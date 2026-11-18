/-
TARGET STATEMENT ONLY — NOT REGISTERED IN `formal/lakefile.lean`.

This file states the stage-1 theorem that the composition in
`Bootstrap.SporosX1Comp` and the real-image instantiation in
`Bootstrap.SporosX1Image` are aimed at.  The proofs are `sorry` on
purpose: the point of the file is to make the remaining subgoals
(parse/decode, refusal branches, termination accounting) reviewable
before more proof work is invested in them.  Do not import this file
from a registered module until the sorries are gone.

ORACLE EXISTENCE: `OracleAdmits` and `¬ HitsNomem` are whole-run
hypotheses of the main targets, which are correct as they stand
(without room, a legal oracle refuses memory and the exit-2 target
applies).  The non-vacuity witness is therefore a separate lemma,
`stage1_oracle_exists_target` below, under an explicit `Room`
hypothesis: a free interval of `stage1Mf f` bytes.  `Stage1InitOk`
and the main statements are unchanged.  What that lemma still
needs is the loader-prefix invariant at the four `mmap`/`mremap`
sites.
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

/-! ### Placement infrastructure for the loader-prefix invariant

The four allocation sites each need three facts: the placement the
witness oracle offers is legal in the memory reached at that site
(`place_at`, fed by `Mem.free` facts transported through appends,
subintervals and erasure), the reads in between do not consume a
witness entry (`kern_count_at_head`), and those two assemble into
`OracleLegalAt` at an mmap/mremap-shaped state
(`oracleLegalAt_mmap`, `oracleLegalAt_mremap`).  What then remains
for `stage1_oracle_exists_target` is reachability: at each site,
the register/region shapes below and the drop count of the witness. -/

theorem place_at {m : Mem} {a n : Nat} (ha : a % page = 0)
    (hpos : 0 < a) (hbound : a + n ≤ 2 ^ 47) (hfree : m.free a n = true) :
    X86Env.place m (some (.at a)) n = some (some a) := by
  simp [X86Env.place, ha, hpos, hbound, hfree]

theorem free_sub {m : Mem} {A n1 n2 : Nat}
    (h : m.free A (n1 + n2) = true) : m.free (A + n1) n2 = true := by
  rw [Mem.free, List.all_eq_true]
  intro r hr
  rcases X86Env.free_spec h r hr with h1 | h1 <;>
    simp only [Bool.or_eq_true, decide_eq_true_eq] <;> omega

theorem free_prefix {m : Mem} {A n1 n2 : Nat}
    (h : m.free A (n1 + n2) = true) : m.free A n1 = true := by
  rw [Mem.free, List.all_eq_true]
  intro r hr
  rcases X86Env.free_spec h r hr with h1 | h1 <;>
    simp only [Bool.or_eq_true, decide_eq_true_eq] <;> omega

theorem free_append {m : Mem} {a n : Nat} {r : Region}
    (h : m.free a n = true) (hdisj : a + n ≤ r.base ∨ r.hi ≤ a) :
    Mem.free (m ++ [r]) a n = true := by
  rw [Mem.free, List.all_eq_true] at h ⊢
  intro s hs
  rcases List.mem_append.1 hs with hs | hs
  · exact h s hs
  · simp at hs
    subst hs
    simp only [Bool.or_eq_true, decide_eq_true_eq]
    exact hdisj

theorem free_eraseIdx {m : Mem} {a n i : Nat}
    (h : m.free a n = true) : Mem.free (m.eraseIdx i) a n = true := by
  rw [Mem.free, List.all_eq_true] at h ⊢
  exact fun s hs => h s (List.mem_of_mem_eraseIdx hs)

/-- A read/write count never consumes a placement entry: with an
`at` head, `Kern.count` moves the full amount and keeps the kernel. -/
theorem kern_count_at_head {k : Kern} {A want avail : Nat}
    (h : k.oracle.head? = some (.at A)) :
    Kern.count k want avail = (min want avail, k) := by
  cases ho : k.oracle with
  | nil => simp [ho] at h
  | cons c rest =>
    simp [ho] at h
    subst h
    simp [Kern.count, Kern.next, ho]

/-- Every entry of the witness oracle is a placement. -/
theorem stage1Oracle_mem_at {f : List Nat} {A : Nat} {c : Choice}
    (hc : c ∈ stage1Oracle f A) : ∃ x, c = .at x := by
  simp [stage1Oracle] at hc
  rcases hc with ⟨-, rfl⟩ | rfl | rfl <;> exact ⟨_, rfl⟩

theorem oracleLegalAt_mmap {s : X86Int.St} {B : Nat}
    (hrip : stage1Im s.rip = some (.syscall, 2))
    (h0 : s.reg 0 = some 9) (h7 : s.reg 7 = some 0)
    {a2 : Nat} (h6 : s.reg 6 = some a2) (h2 : s.reg 2 = some PROT_RW)
    (h10 : s.reg 10 = some MAP_PRIVATE_ANON)
    (h8 : s.reg 8 = some (W - 1)) (h9 : s.reg 9 = some 0)
    (ha2 : a2 ≠ 0) (hhead : s.k.oracle.head? = some (.at B))
    (hplace : X86Env.place s.mem (some (.at B)) (X86Env.pages a2) =
      some (some B)) :
    OracleLegalAt s := by
  intro _hmem
  obtain ⟨rax, k', m', hsc⟩ :=
    syscall_mmap_ret_of_place ha2 hhead hplace
  exact ⟨9, 0, a2, PROT_RW, MAP_PRIVATE_ANON, W - 1, 0, rax, k', m',
    h0, h7, h6, h2, h10, h8, h9, hsc⟩

theorem oracleLegalAt_mremap {s : X86Int.St} {B : Nat}
    (hrip : stage1Im s.rip = some (.syscall, 2))
    (h0 : s.reg 0 = some 25)
    {a1 a2 a3 : Nat} (h7 : s.reg 7 = some a1) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some MREMAP_MAYMOVE)
    (h8 : s.reg 8 = some 0) (h9 : s.reg 9 = some 0)
    {i : Nat} {r : Region}
    (hfind : s.mem.find a1 0 = some (i, r))
    (hbase : r.base = a1) (hlen : r.data.length = X86Env.pages a2)
    (hrw : r.rw = true) (hgrow : ¬ X86Env.pages a3 < r.data.length)
    (hhead : s.k.oracle.head? = some (.at B))
    (hplace : X86Env.place (s.mem.eraseIdx i) (some (.at B))
      (X86Env.pages a3) = some (some B)) :
    OracleLegalAt s := by
  intro _hmem
  obtain ⟨rax, k', m', hsc⟩ :=
    syscall_mremap_ret_of_place hfind hbase hlen hrw hgrow hhead hplace
  exact ⟨25, a1, a2, a3, MREMAP_MAYMOVE, 0, 0, rax, k', m',
    h0, h7, h6, h2, h10, h8, h9, hsc⟩

/-! ### The loader prefix layout (entry to `loaded`, plus the fail blocks)

Offsets are the listing's.  The two memory syscalls of this region
are the initial file-buffer `mmap` at `base + 0xbf` and the growth
`mremap` at `base + 0x113`; the syscalls at `0x92`, `0xe8`, `0x12a`
and `0x132` are `open`, `read` and the two exits, none of which is a
memory syscall. -/

structure LoaderCode (im : Nat → Option (Ins × Nat)) : Prop where
  d78 : im (base + 0x78) = some (.pop 0, 1)
  d79 : im (base + 0x79) = some (.cmpI 0 2, 4)
  d7d : im (base + 0x7d) = some (.jcc 13 (base + 0x84), 2)
  d7f : im (base + 0x7f) = some (.jmp (base + 0x124), 5)
  d84 : im (base + 0x84) = some (.pop 6, 1)
  d85 : im (base + 0x85) = some (.pop 6, 1)
  d86 : im (base + 0x86) = some (.movI 0 2, 5)
  d8b : im (base + 0x8b) = some (.mov 7 6, 3)
  d8e : im (base + 0x8e) = some (.alu .xor 6 6, 2)
  d90 : im (base + 0x90) = some (.alu .xor 2 2, 2)
  d92 : im (base + 0x92) = some (.syscall, 2)
  d94 : im (base + 0x94) = some (.test 0 0, 3)
  d97 : im (base + 0x97) = some (.jcc 9 (base + 0x9e), 2)
  d99 : im (base + 0x99) = some (.jmp (base + 0x124), 5)
  d9e : im (base + 0x9e) = some (.mov 14 0, 3)
  da1 : im (base + 0xa1) = some (.movI 0 9, 5)
  da6 : im (base + 0xa6) = some (.alu .xor 7 7, 2)
  da8 : im (base + 0xa8) = some (.movI 6 65536, 5)
  dad : im (base + 0xad) = some (.movI 2 3, 5)
  db2 : im (base + 0xb2) = some (.movI 10 34, 6)
  db8 : im (base + 0xb8) = some (.pushI (W - 1), 2)
  dba : im (base + 0xba) = some (.pop 8, 2)
  dbc : im (base + 0xbc) = some (.alu .xor 9 9, 3)
  dbf : im (base + 0xbf) = some (.syscall, 2)
  dc1 : im (base + 0xc1) = some (.test 0 0, 3)
  dc4 : im (base + 0xc4) = some (.jcc 9 (base + 0xcb), 2)
  dc6 : im (base + 0xc6) = some (.jmp (base + 0x12c), 5)
  dcb : im (base + 0xcb) = some (.mov 3 0, 3)
  dce : im (base + 0xce) = some (.movI 12 65536, 6)
  dd4 : im (base + 0xd4) = some (.alu .xor 13 13, 3)
  dd7 : im (base + 0xd7) = some (.mov 6 3, 3)
  dda : im (base + 0xda) = some (.alu .add 6 13, 3)
  ddd : im (base + 0xdd) = some (.mov 2 12, 3)
  de0 : im (base + 0xe0) = some (.alu .sub 2 13, 3)
  de3 : im (base + 0xe3) = some (.alu .xor 0 0, 2)
  de5 : im (base + 0xe5) = some (.mov32 7 14, 3)
  de8 : im (base + 0xe8) = some (.syscall, 2)
  dea : im (base + 0xea) = some (.test 0 0, 3)
  ded : im (base + 0xed) = some (.jcc 8 (base + 0x124), 2)
  def_ : im (base + 0xef) = some (.jcc 4 (base + 0x122), 2)
  df1 : im (base + 0xf1) = some (.alu .add 13 0, 3)
  df4 : im (base + 0xf4) = some (.mov 2 12, 3)
  df7 : im (base + 0xf7) = some (.alu .sub 2 13, 3)
  dfa : im (base + 0xfa) = some (.jcc 5 (base + 0xd7), 2)
  dfc : im (base + 0xfc) = some (.movI 0 25, 5)
  d101 : im (base + 0x101) = some (.mov 7 3, 3)
  d104 : im (base + 0x104) = some (.mov 6 12, 3)
  d107 : im (base + 0x107) = some (.mov 2 12, 3)
  d10a : im (base + 0x10a) = some (.alu .add 2 2, 3)
  d10d : im (base + 0x10d) = some (.movI 10 1, 6)
  d113 : im (base + 0x113) = some (.syscall, 2)
  d115 : im (base + 0x115) = some (.test 0 0, 3)
  d118 : im (base + 0x118) = some (.jcc 8 (base + 0x12c), 2)
  d11a : im (base + 0x11a) = some (.mov 3 0, 3)
  d11d : im (base + 0x11d) = some (.mov 12 2, 3)
  d120 : im (base + 0x120) = some (.jmp (base + 0xd7), 2)
  d122 : im (base + 0x122) = some (.jmp (base + 0x134), 2)
  d124 : im (base + 0x124) = some (.pushI 60, 2)
  d126 : im (base + 0x126) = some (.pop 0, 1)
  d127 : im (base + 0x127) = some (.pushI 1, 2)
  d129 : im (base + 0x129) = some (.pop 7, 1)
  d12a : im (base + 0x12a) = some (.syscall, 2)
  d12c : im (base + 0x12c) = some (.pushI 60, 2)
  d12e : im (base + 0x12e) = some (.pop 0, 1)
  d12f : im (base + 0x12f) = some (.pushI 2, 2)
  d131 : im (base + 0x131) = some (.pop 7, 1)
  d132 : im (base + 0x132) = some (.syscall, 2)

theorem stage1_loaderCode : LoaderCode stage1Im := by
  apply LoaderCode.mk <;> decide +kernel

/-! ### Initial-stack facts for the loader prefix

The three `pop`s at entry read `argc` and the two `argv` pointers
from the stack the kernel built.  These lemmas discharge those
reads against `X86Env.initStack` for a two-string `argv`, in the
two-region initial memory: the bytes from `sp` are the seven head
words followed by the two NUL-terminated strings. -/

theorem pages_ge (n : Nat) : n ≤ X86Env.pages n := by
  unfold X86Env.pages X86Env.page
  have h := Nat.div_add_mod (n + 4096 - 1) 4096
  have hlt := Nat.mod_lt (n + 4096 - 1) (by decide : 0 < 4096)
  omega

theorem le64_length (v : Nat) : (X86Env.le64 v).length = 8 := rfl

theorem getElem?_append_of_lt {l1 l2 : List Nat} {n : Nat}
    (h : n < l1.length) : (l1 ++ l2)[n]? = l1[n]? := by
  induction l1 generalizing n with
  | nil => simp at h
  | cons a t ih =>
    cases n with
    | zero => rfl
    | succ k => exact ih (by simpa using h)

theorem getElem?_replicate_append (n j : Nat) {bs : List Nat} :
    (List.replicate n 0 ++ bs)[n + j]? = bs[j]? := by
  induction n with
  | zero => simp
  | succ k ih =>
    rw [List.replicate_succ, List.cons_append,
      show k + 1 + j = (k + j) + 1 from by omega]
    exact ih

set_option maxHeartbeats 400000 in
/-- The bytes the initial stack holds from `sp`: argc, the two argv
pointers, four zero words, then the two NUL-terminated strings. -/
theorem initStack_data_at (top : Nat) (x y : List Nat) {j : Nat}
    (hj : j < 56 + (x.length + 1) + (y.length + 1)) :
    (X86Env.initStack top [x, y]).1.data[X86Env.stackRoom + j]? =
      (X86Env.le64 2 ++ (X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56) ++
        (X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56 + x.length + 1) ++
        (X86Env.le64 0 ++ (X86Env.le64 0 ++ (X86Env.le64 0 ++ (X86Env.le64 0 ++
        (x ++ ([0] ++ (y ++ [0]))))))))))[j]? := by
  have hsp : (X86Env.initStack top [x, y]).2 =
      (X86Env.initStack top [x, y]).1.base + X86Env.stackRoom := by
    simp only [X86Env.initStack]
  rw [hsp]
  simp only [X86Env.initStack, X86Env.layStrings, List.flatten_cons,
    List.map_cons, List.map_nil, List.flatten_nil, List.append_nil]
  rw [getElem?_append_of_lt (by simp [le64_length]; omega)]
  rw [getElem?_replicate_append]
  simp only [List.append_assoc]
  congr 2 <;> simp

theorem getElem?_append_skip {l1 l2 : List Nat} {i : Nat} :
    (l1 ++ l2)[l1.length + i]? = l2[i]? := by
  induction l1 with
  | nil => simp
  | cons a t ih =>
    simp only [List.length_cons, List.cons_append]
    rw [show t.length + 1 + i = (t.length + i) + 1 from by omega]
    exact ih

theorem mem_rd_second {c r : Region} {a v : Nat}
    (hdisj : c.hi ≤ r.base) (hge : r.base ≤ a)
    (hbyte : r.data[a - r.base]? = some v) :
    Mem.rd [c, r] a = some v := by
  have hlt : a < r.hi := by
    have h1 := (List.getElem?_eq_some_iff.1 hbyte).1
    unfold Region.hi
    omega
  have hnot : ¬ (c.base ≤ a ∧ a < c.hi) := by omega
  simp [Mem.rd, Mem.find, hnot, hge, hlt, hbyte]

theorem rdN_of_rd {m : Mem} {a n : Nat} {L : List Nat} (hL : L.length = n)
    (h : ∀ i (hi : i < n), ∃ v, m.rd (a + i) = some v ∧ L[i]? = some v) :
    m.rdN a n = some L := by
  induction n generalizing a L with
  | zero =>
    have : L = [] := by cases L <;> simp_all
    simp [Mem.rdN, this]
  | succ k ih =>
    cases L with
    | nil => simp at hL
    | cons b t =>
      have ht : t.length = k := by simpa using hL
      obtain ⟨v, hv, hb⟩ := h 0 (by omega)
      simp at hb
      subst hb
      have hrec : Mem.rdN m (a + 1) k = some t := by
        apply ih ht
        intro i hi
        obtain ⟨w, hw, htw⟩ := h (i + 1) (by omega)
        refine ⟨w, ?_, ?_⟩
        · rwa [show a + 1 + i = a + (i + 1) from by omega]
        · simpa using htw
      simp only [Mem.rdN]
      rw [show m.rd a = some b from by simpa using hv, hrec]
      rfl

set_option maxHeartbeats 800000 in
theorem leVal_le64 (v : Nat) : leVal (X86Env.le64 v) = v % W := by
  have h1 : X86Env.le64 v = bytes8 v := rfl
  rw [h1]
  have hd := leVal_digits 8 v
  have hw : 256 ^ 8 = W := rfl
  rwa [hw] at hd

set_option maxHeartbeats 800000 in
theorem rd64_second_of_bytes {c r : Region} {a : Nat} {L : List Nat}
    (hdisj : c.hi ≤ r.base) (hge : r.base ≤ a) (hL : L.length = 8)
    (h : ∀ i (hi : i < 8), r.data[a - r.base + i]? = L[i]?) :
    rd64 [c, r] a = some (leVal L) := by
  unfold rd64
  rw [rdN_of_rd hL]
  · simp
  · intro i hi
    have hb := h i hi
    obtain ⟨v, hv⟩ : ∃ v, L[i]? = some v := by
      cases hLi : L[i]? with
      | some v => exact ⟨v, rfl⟩
      | none =>
        simp [hL] at hLi
        omega
    refine ⟨v, ?_, hv⟩
    apply mem_rd_second hdisj (by omega)
    rwa [show a + i - r.base = a - r.base + i from by omega, hb]

theorem initStack_sp (top : Nat) (x y : List Nat) :
    (X86Env.initStack top [x, y]).2 =
      (X86Env.initStack top [x, y]).1.base + X86Env.stackRoom := by
  simp only [X86Env.initStack]

theorem initStack_data_length (top : Nat) (x y : List Nat) :
    (X86Env.initStack top [x, y]).1.data.length =
      X86Env.stackRoom +
        X86Env.pages (56 + ((x.length + 1) + (y.length + 1)) + 1) := by
  have hneed : 56 + ((x.length + 1) + (y.length + 1)) ≤
      X86Env.pages (56 + ((x.length + 1) + (y.length + 1)) + 1) :=
    Nat.le_trans (Nat.le_succ _) (pages_ge _)
  simp only [X86Env.initStack]
  simp [X86Env.layStrings, le64_length]
  omega

theorem initStack_rd64_argc {c : Region} {top : Nat} {x y : List Nat}
    (hdisj : c.hi ≤ (X86Env.initStack top [x, y]).1.base) :
    rd64 [c, (X86Env.initStack top [x, y]).1]
      ((X86Env.initStack top [x, y]).2) = some 2 := by
  have hsp := initStack_sp top x y
  have key := rd64_second_of_bytes (L := X86Env.le64 2) hdisj (by omega)
    (le64_length _) (a := (X86Env.initStack top [x, y]).2) (by
      intro i hi
      rw [hsp, Nat.add_sub_cancel_left, initStack_data_at top x y (by omega)]
      exact getElem?_append_of_lt (by rw [le64_length]; exact hi))
  rwa [leVal_le64] at key

theorem initStack_rd64_argv0 {c : Region} {top : Nat} {x y : List Nat}
    (hdisj : c.hi ≤ (X86Env.initStack top [x, y]).1.base) :
    rd64 [c, (X86Env.initStack top [x, y]).1]
        ((X86Env.initStack top [x, y]).2 + 8) =
      some (((X86Env.initStack top [x, y]).2 + 56) % W) := by
  have hsp := initStack_sp top x y
  have key := rd64_second_of_bytes
    (L := X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56)) hdisj
    (by omega) (le64_length _) (a := (X86Env.initStack top [x, y]).2 + 8) (by
      intro i hi
      rw [show (X86Env.initStack top [x, y]).2 + 8 -
            (X86Env.initStack top [x, y]).1.base + i =
          X86Env.stackRoom + (8 + i) from by omega,
        initStack_data_at top x y (by omega),
        show 8 + i = (X86Env.le64 2).length + i from by rw [le64_length],
        getElem?_append_skip]
      exact getElem?_append_of_lt (by rw [le64_length]; exact hi))
  rwa [leVal_le64] at key

theorem initStack_rd64_argv1 {c : Region} {top : Nat} {x y : List Nat}
    (hdisj : c.hi ≤ (X86Env.initStack top [x, y]).1.base) :
    rd64 [c, (X86Env.initStack top [x, y]).1]
        ((X86Env.initStack top [x, y]).2 + 16) =
      some (((X86Env.initStack top [x, y]).2 + 56 + x.length + 1) % W) := by
  have hsp := initStack_sp top x y
  have key := rd64_second_of_bytes
    (L := X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56 + x.length + 1))
    hdisj (by omega) (le64_length _)
    (a := (X86Env.initStack top [x, y]).2 + 16) (by
      intro i hi
      rw [show (X86Env.initStack top [x, y]).2 + 16 -
            (X86Env.initStack top [x, y]).1.base + i =
          X86Env.stackRoom + (16 + i) from by omega,
        initStack_data_at top x y (by omega),
        show 16 + i = (X86Env.le64 2).length + (8 + i) from by
          have h8 := le64_length 2; omega,
        getElem?_append_skip,
        show 8 + i =
          (X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56)).length + i from by
          rw [le64_length],
        getElem?_append_skip]
      exact getElem?_append_of_lt (by rw [le64_length]; exact hi))
  rwa [leVal_le64] at key

theorem cstr_of_bytes {m : Mem} {a n : Nat} {s : List Nat}
    (hn : s.length < n)
    (h : ∀ i (hi : i < s.length), ∃ v, m.rd (a + i) = some v ∧ s[i]? = some v)
    (h0 : ∀ b ∈ s, b ≠ 0) (hend : m.rd (a + s.length) = some 0) :
    m.cstr a n = some s := by
  induction s generalizing a n with
  | nil =>
    cases n with
    | zero => simp at hn
    | succ k =>
      simp only [List.length_nil, Nat.add_zero] at hend
      simp [Mem.cstr, hend]
  | cons b t ih =>
    cases n with
    | zero => simp at hn
    | succ k =>
      obtain ⟨v, hv, hb⟩ := h 0 (by simp)
      simp at hb
      subst hb
      have hb0 : b ≠ 0 := h0 b (by simp)
      have hrec : m.cstr (a + 1) k = some t := by
        apply ih (by simpa using hn)
        · intro i hi
          obtain ⟨w, hw, htw⟩ := h (i + 1) (by simpa using hi)
          refine ⟨w, ?_, ?_⟩
          · rwa [show a + 1 + i = a + (i + 1) from by omega]
          · simpa using htw
        · exact fun z hz => h0 z (List.mem_cons_of_mem b hz)
        · rwa [show a + 1 + t.length = a + (b :: t).length from by simp; omega]
      simp only [Mem.cstr]
      rw [show m.rd a = some b from by simpa using hv]
      simp [hb0, hrec]

theorem initStack_rd_path {c : Region} {top : Nat} {x y : List Nat}
    (hdisj : c.hi ≤ (X86Env.initStack top [x, y]).1.base) {i : Nat}
    (hi : i ≤ y.length) :
    Mem.rd [c, (X86Env.initStack top [x, y]).1]
        ((X86Env.initStack top [x, y]).2 + 56 + x.length + 1 + i) =
      (y ++ [0])[i]? := by
  have hsp := initStack_sp top x y
  obtain ⟨v, hv⟩ : ∃ v, (y ++ [0])[i]? = some v := by
    cases hLi : (y ++ [0])[i]? with
    | some v => exact ⟨v, rfl⟩
    | none =>
      simp [List.getElem?_eq_none_iff] at hLi
      omega
  rw [hv]
  apply mem_rd_second hdisj (by omega)
  rw [show (X86Env.initStack top [x, y]).2 + 56 + x.length + 1 + i -
        (X86Env.initStack top [x, y]).1.base =
      X86Env.stackRoom + (56 + x.length + 1 + i) from by omega,
    initStack_data_at top x y (by omega)]
  rw [show 56 + x.length + 1 + i = (X86Env.le64 2).length + (48 + x.length + 1 + i)
      from by have h8 := le64_length 2; omega, getElem?_append_skip,
    show 48 + x.length + 1 + i =
        (X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56)).length +
          (40 + x.length + 1 + i)
      from by have h8 := le64_length ((X86Env.initStack top [x, y]).2 + 56); omega, getElem?_append_skip,
    show 40 + x.length + 1 + i =
        (X86Env.le64 ((X86Env.initStack top [x, y]).2 + 56 + x.length + 1)).length +
          (32 + x.length + 1 + i)
      from by have h8 := le64_length ((X86Env.initStack top [x, y]).2 + 56 + x.length + 1); omega, getElem?_append_skip,
    show 32 + x.length + 1 + i = (X86Env.le64 0).length + (24 + x.length + 1 + i)
      from by have h8 := le64_length 0; omega, getElem?_append_skip,
    show 24 + x.length + 1 + i = (X86Env.le64 0).length + (16 + x.length + 1 + i)
      from by have h8 := le64_length 0; omega, getElem?_append_skip,
    show 16 + x.length + 1 + i = (X86Env.le64 0).length + (8 + x.length + 1 + i)
      from by have h8 := le64_length 0; omega, getElem?_append_skip,
    show 8 + x.length + 1 + i = (X86Env.le64 0).length + (x.length + 1 + i)
      from by have h8 := le64_length 0; omega, getElem?_append_skip,
    show x.length + 1 + i = x.length + (1 + i) from by omega,
    getElem?_append_skip,
    show 1 + i = [0].length + i from rfl, getElem?_append_skip]
  exact hv

theorem initStack_cstr_path {c : Region} {top : Nat} {x y : List Nat}
    (hdisj : c.hi ≤ (X86Env.initStack top [x, y]).1.base)
    (h0 : 0 ∉ y) (hlen : y.length < 4096) :
    Mem.cstr [c, (X86Env.initStack top [x, y]).1]
        ((X86Env.initStack top [x, y]).2 + 56 + x.length + 1) 4096 =
      some y := by
  apply cstr_of_bytes hlen
  · intro i hi
    have hb := initStack_rd_path hdisj (i := i) (by omega)
    obtain ⟨v, hv⟩ : ∃ v, y[i]? = some v := by
      cases hLi : y[i]? with
      | some v => exact ⟨v, rfl⟩
      | none =>
        simp [List.getElem?_eq_none_iff] at hLi
        omega
    refine ⟨v, ?_, hv⟩
    have hsplit : (y ++ [0])[i]? = y[i]? :=
      getElem?_append_of_lt hi
    rwa [hsplit, hv] at hb
  · intro b hb hb0
    exact h0 (hb0 ▸ hb)
  · have hb := initStack_rd_path hdisj (i := y.length) (Nat.le_refl _)
    rw [show (y ++ [0])[y.length]? = some 0 from by
      rw [show y.length = y.length + 0 from by omega, getElem?_append_skip]
      rfl] at hb
    exact hb

/-! ### Entry to the initial `mmap`: step infrastructure -/

theorem step_syscall_ret {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {len nr a1 a2 a3 a4 a5 a6 rax : Nat} {k' : Kern} {m' : Mem}
    (hi : im s.rip = some (.syscall, len)) (h0 : s.reg 0 = some nr)
    (h7 : s.reg 7 = some a1) (h6 : s.reg 6 = some a2)
    (h2 : s.reg 2 = some a3) (h10 : s.reg 10 = some a4)
    (h8 : s.reg 8 = some a5) (h9 : s.reg 9 = some a6)
    (hsc : X86Env.syscall s.k s.mem nr a1 a2 a3 a4 a5 a6 =
      some (.ret rax k' m')) :
    X86Int.step im s =
      .ok { ((s.set 0 (some rax)).set 1 (some (s.rip + len))).set 11 none with
        k := k', mem := m', rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, h0, h7, h6, h2, h10, h8, h9, hsc]

theorem free_of_wshape {m m' : Mem} (h : wshape m' = wshape m)
    {a n : Nat} : m'.free a n = m.free a n := by
  induction m generalizing m' with
  | nil =>
    cases m' with
    | nil => rfl
    | cons r t => simp [wshape] at h
  | cons r t ih =>
    cases m' with
    | nil => simp [wshape] at h
    | cons r' t' =>
      have h1 : (r'.base, r'.hi, r'.rw) = (r.base, r.hi, r.rw) := by
        have := List.cons.inj h
        exact this.1
      have h2 : wshape t' = wshape t := (List.cons.inj h).2
      simp only [Mem.free, List.all_cons]
      have ih' : t'.all (fun r => a + n ≤ r.base || r.hi ≤ a) =
          t.all (fun r => a + n ≤ r.base || r.hi ≤ a) := ih h2
      rw [ih']
      have hb : r'.base = r.base := by
        have := congrArg Prod.fst h1
        simpa using this
      have hh : r'.hi = r.hi := by
        have := congrArg (fun p => p.2.1) h1
        simpa using this
      rw [hb, hh]

theorem mem_wr_second {c r : Region} {a v : Nat}
    (hdisj : c.hi ≤ r.base) (hge : r.base ≤ a) (hlt : a < r.hi)
    (hrw : r.rw = true) :
    Mem.wr [c, r] a v =
      some [c, { r with data := r.data.set (a - r.base) (v % 256) }] := by
  have hnot : ¬ (c.base ≤ a ∧ a < c.hi) := by omega
  simp [Mem.wr, Mem.find, hnot, hge, hlt, hrw]

theorem wrN_second_some : ∀ (bs : List Nat) {c r : Region} {a : Nat},
    c.hi ≤ r.base → r.rw = true → r.base ≤ a → a + bs.length ≤ r.hi →
    ∃ r', Mem.wrN [c, r] a bs = some [c, r'] ∧ r'.base = r.base ∧
      r'.hi = r.hi ∧ r'.rw = r.rw
  | [], c, r, a, _, _, _, _ => ⟨r, by simp [Mem.wrN], rfl, rfl, rfl⟩
  | b :: t, c, r, a, hdisj, hrw, hge, hhi => by
    have hlt : a < r.hi := by
      simp at hhi
      omega
    rw [Mem.wrN, mem_wr_second hdisj hge hlt hrw]
    have hlen : ({ r with data := r.data.set (a - r.base) (b % 256) } :
        Region).hi = r.hi := by
      simp [Region.hi, List.length_set]
    obtain ⟨r'', hr'', hb, hh, hrw'⟩ :=
      wrN_second_some (c := c)
        (r := { r with data := r.data.set (a - r.base) (b % 256) })
        (a := a + 1) t hdisj (by simpa using hrw) (by show r.base ≤ a + 1; omega) (by
        rw [hlen]
        simp at hhi
        omega)
    exact ⟨r'', by simpa using hr'', by simpa using hb,
      by rw [hh, hlen], by simpa using hrw'⟩

theorem stage1Init_rip (f path argv0 stdin : List Nat)
    (oracle : List Choice) (top : Nat) :
    (stage1Init f path argv0 stdin oracle top).rip = base + 0x78 := by
  simp [stage1Init, elfEntry_eq]

theorem stage1Init_mem (f path argv0 stdin : List Nat)
    (oracle : List Choice) (top : Nat) :
    (stage1Init f path argv0 stdin oracle top).mem =
      [⟨base, stage1Code, false⟩,
        (X86Env.initStack top [argv0, path]).1] := by
  simp [stage1Init, phVaddr_eq, loadSegment_eq]

theorem stage1Init_reg4 (f path argv0 stdin : List Nat)
    (oracle : List Choice) (top : Nat) :
    (stage1Init f path argv0 stdin oracle top).reg 4 =
      some (X86Env.initStack top [argv0, path]).2 := by
  simp [stage1Init, St.reg]

theorem stage1Init_reg_ne4 (f path argv0 stdin : List Nat)
    (oracle : List Choice) (top : Nat) {i : Nat} (hi : i < 16)
    (h4 : i ≠ 4) :
    (stage1Init f path argv0 stdin oracle top).reg i = some 0 := by
  simp only [stage1Init, St.reg, List.getElem?_set]
  rw [if_neg (by omega : ¬ (4 = i))]
  have hrep : (List.replicate 16 (some 0))[i]? = some (some 0) := by
    rw [List.getElem?_replicate]
    simp [hi]
  rw [hrep]
  rfl

theorem stage1Init_regs_length (f path argv0 stdin : List Nat)
    (oracle : List Choice) (top : Nat) :
    (stage1Init f path argv0 stdin oracle top).regs.length = 16 := by
  simp [stage1Init]

theorem cond_cc13_of_sf_eq_of {f : Flags} {b : Bool}
    (hsf : f.sf = some b) (hof : f.of = some b) :
    X86Int.cond f 13 = some true := by
  simp [X86Int.cond, hsf, hof]

theorem cond_cc9_of_sf_false {f : Flags} (hsf : f.sf = some false) :
    X86Int.cond f 9 = some true := by
  simp [X86Int.cond, hsf]

theorem cond_cc8_of_sf_false {f : Flags} (hsf : f.sf = some false) :
    X86Int.cond f 8 = some false := by
  simp [X86Int.cond, hsf]

theorem find_some_mem : ∀ {m : Mem} {a k i : Nat} {r : Region},
    Mem.find m a k = some (i, r) → r ∈ m ∧ r.base ≤ a ∧ a < r.hi
  | [], _, _, _, _, h => by simp [Mem.find] at h
  | r :: t, a, k, i, r', h => by
    unfold Mem.find at h
    by_cases hc : r.base ≤ a ∧ a < r.hi
    · rw [if_pos hc] at h
      simp only [Option.some.injEq, Prod.mk.injEq] at h
      obtain ⟨-, rfl⟩ := h
      exact ⟨List.mem_cons_self, hc.1, hc.2⟩
    · rw [if_neg hc] at h
      obtain ⟨h1, h2, h3⟩ := find_some_mem h
      exact ⟨List.mem_cons_of_mem r h1, h2, h3⟩

theorem rd_some_region {m : Mem} {a v : Nat}
    (h : Mem.rd m a = some v) : ∃ r ∈ m, r.base ≤ a ∧ a < r.hi := by
  unfold Mem.rd at h
  cases hfind : Mem.find m a 0 with
  | none => simp [hfind] at h
  | some ir =>
    obtain ⟨i, r⟩ := ir
    obtain ⟨h1, h2, h3⟩ := find_some_mem hfind
    exact ⟨r, h1, h2, h3⟩

/-- Entry to the initial `mmap` site: 21 steps.  From a
loader-shaped state the run reaches `base + 0xbf` with the exact
`mmap` argument registers, fd 3 open on the file, the oracle
untouched (neither `open` nor the arithmetic consumes a choice),
and memory of the same shape (the one `pushI` writes inside the
stack region). -/
theorem loader_entry_to_mmap {f path argv0 : List Nat} {oracle : List Choice}
    {top sp0 : Nat} {stdinL : List Nat} {stk : Region} {s : X86Int.St}
    (hstk : stk = (X86Env.initStack top [argv0, path]).1)
    (hsp0 : sp0 = (X86Env.initStack top [argv0, path]).2)
    (hrip : s.rip = base + 0x78) (hlen16 : s.regs.length = 16)
    (hr4 : s.reg 4 = some sp0)
    (hr0 : ∀ i, i ≠ 4 → i < 16 → s.reg i = some 0)
    (hmem : s.mem = [⟨base, stage1Code, false⟩, stk])
    (hk : s.k = ⟨[(path, f)], [], stdinL, [], oracle⟩)
    (hdisj : base + stage1Code.length ≤ stk.base)
    (hstkhi : stk.hi = top) (htop47 : top ≤ 2 ^ 47)
    (hpath0 : 0 ∉ path) (hpathlen : path.length < 4096) :
    ∃ s', run stage1Im 21 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0xbf ∧
      s'.reg 0 = some 9 ∧ s'.reg 7 = some 0 ∧ s'.reg 6 = some 65536 ∧
      s'.reg 2 = some 3 ∧ s'.reg 10 = some 34 ∧
      s'.reg 8 = some (W - 1) ∧ s'.reg 9 = some 0 ∧
      s'.reg 14 = some 3 ∧ s'.reg 4 = some (sp0 + 24) ∧
      s'.k = ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ ∧
      wshape s'.mem = wshape s.mem ∧ s'.regs.length = 16 := by
  have hc := stage1_loaderCode
  have hSR : X86Env.stackRoom = 4096 := rfl
  have hsp0eq : sp0 = stk.base + X86Env.stackRoom := by
    rw [hsp0, hstk]
    exact initStack_sp top argv0 path
  have hdatalen : stk.data.length =
      X86Env.stackRoom +
        X86Env.pages (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
    rw [hstk]
    exact initStack_data_length top argv0 path
  have hsp0le : sp0 ≤ top := by
    have h1 : stk.base + X86Env.stackRoom ≤ stk.hi := by
      unfold Region.hi
      omega
    omega
  have hspW : sp0 + 24 < W := by
    unfold W
    omega
  have hdisj' : (⟨base, stage1Code, false⟩ : Region).hi ≤ stk.base := by
    simp only [Region.hi]
    exact hdisj
  -- the path pointer and its bound
  have hP : sp0 + 56 + argv0.length + 1 < W := by
    have hb := initStack_rd_path (c := ⟨base, stage1Code, false⟩)
      (top := top) (x := argv0) (y := path) (by rw [← hstk]; exact hdisj')
      (i := 0) (by omega)
    rw [← hstk, ← hsp0] at hb
    obtain ⟨v, hv⟩ : ∃ v, Mem.rd [⟨base, stage1Code, false⟩, stk]
        (sp0 + 56 + argv0.length + 1 + 0) = some v := by
      cases hLi : (path ++ [0])[0]? with
      | some v => exact ⟨v, by rwa [hLi] at hb⟩
      | none =>
        cases path with
        | nil => simp at hLi
        | cons b t => simp at hLi
    obtain ⟨r, hr, -, hlt⟩ := rd_some_region hv
    have hr2 : r = ⟨base, stage1Code, false⟩ ∨ r = stk := by simpa using hr
    have hrhi : r.hi ≤ top := by
      rcases hr2 with rfl | rfl
      · have h1 : stk.base ≤ stk.hi := by
          unfold Region.hi
          omega
        omega
      · omega
    unfold W
    omega
  have hPmod : (sp0 + 56 + argv0.length + 1) % W =
      sp0 + 56 + argv0.length + 1 := Nat.mod_eq_of_lt hP
  have hA0mod : (sp0 + 56) % W = sp0 + 56 :=
    Nat.mod_eq_of_lt (by omega)
  -- step 1: pop rax (argc = 2)
  have hrd1 : rd64 s.mem sp0 = some 2 := by
    have h := initStack_rd64_argc (by rw [← hstk]; exact hdisj')
    rw [← hstk, ← hsp0] at h
    rwa [hmem]
  have e1 : X86Int.step stage1Im s =
      .ok { (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
        rip := s.rip + 1 } :=
    step_pop (by rw [hrip]; exact hc.d78) hr4 hrd1
  obtain ⟨a1, ha1⟩ : ∃ t : X86Int.St, t =
      { (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
        rip := s.rip + 1 } := ⟨_, rfl⟩
  have ha1rip : a1.rip = base + 0x79 := by
    rw [ha1]; show s.rip + 1 = base + 0x79; rw [hrip]
  have hlen1 : a1.regs.length = 16 := by
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen16
  have ha1r0 : a1.reg 0 = some 2 := by
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 0 = some 2
    rw [reg_with_rip, reg_set_self (by
      show 0 < (s.set 4 (some ((sp0 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ha1r4 : a1.reg 4 = some (sp0 + 8) := by
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 4 = some (sp0 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha1mem : a1.mem = s.mem := by rw [ha1]; rfl
  have ha1k : a1.k = s.k := by rw [ha1]; rfl
  -- step 2: cmp rax, 2
  have e2 : X86Int.step stage1Im a1 =
      .ok { a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 } :=
    step_cmpI (by rw [ha1rip]; exact hc.d79) ha1r0
  obtain ⟨a2, ha2⟩ : ∃ t : X86Int.St, t =
      { a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 } := ⟨_, rfl⟩
  have ha2rip : a2.rip = base + 0x7d := by
    rw [ha2]; show a1.rip + 4 = base + 0x7d; rw [ha1rip]
  have ha2fl : X86Int.cond a2.fl 13 = some true := by
    have h2W : (2 : Nat) % W = 2 := by decide
    rw [ha2]
    show X86Int.cond (aluF .sub 2 (2 % W)).2 13 = some true
    rw [h2W]
    apply cond_cc13_of_sf_eq_of (b := false) <;> decide
  have ha2r4 : a2.reg 4 = some (sp0 + 8) := by rw [ha2]; exact ha1r4
  have ha2mem : a2.mem = s.mem := by rw [ha2]; exact ha1mem
  have ha2k : a2.k = s.k := by rw [ha2]; exact ha1k
  -- step 3: jge 0x84 (taken)
  have e3 : X86Int.step stage1Im a2 = .ok { a2 with rip := base + 0x84 } := by
    have h := step_jcc (im := stage1Im) (s := a2)
      (by rw [ha2rip]; exact hc.d7d) ha2fl
    simpa using h
  obtain ⟨a3, ha3⟩ : ∃ t : X86Int.St, t =
      { a2 with rip := base + 0x84 } := ⟨_, rfl⟩
  have ha3rip : a3.rip = base + 0x84 := by rw [ha3]
  have ha3r4 : a3.reg 4 = some (sp0 + 8) := by
    rw [ha3]; show ({ a2 with rip := base + 0x84 }).reg 4 = some (sp0 + 8)
    rw [reg_with_rip]; exact ha2r4
  have ha3mem : a3.mem = s.mem := by rw [ha3]; exact ha2mem
  have ha3k : a3.k = s.k := by rw [ha3]; exact ha2k
  have hlen3 : a3.regs.length = 16 := by
    rw [ha3]; show a2.regs.length = 16
    rw [ha2]; show a1.regs.length = 16
    exact hlen1
  -- steps 4-5: pop rsi twice (argv[0] discarded, argv[1] kept)
  have hrd4 : rd64 a3.mem (sp0 + 8) = some (sp0 + 56) := by
    have h := initStack_rd64_argv0 (by rw [← hstk]; exact hdisj')
    rw [← hstk, ← hsp0, hA0mod] at h
    rwa [ha3mem, hmem]
  have e4 : X86Int.step stage1Im a3 =
      .ok { (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
        rip := a3.rip + 1 } :=
    step_pop (by rw [ha3rip]; exact hc.d84) ha3r4 hrd4
  obtain ⟨a4, ha4⟩ : ∃ t : X86Int.St, t =
      { (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
        rip := a3.rip + 1 } := ⟨_, rfl⟩
  have ha4rip : a4.rip = base + 0x85 := by
    rw [ha4]; show a3.rip + 1 = base + 0x85; rw [ha3rip]
  have hlen4 : a4.regs.length = 16 := by
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen3
  have ha4r4 : a4.reg 4 = some (sp0 + 16) := by
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 4 = some (sp0 + 16)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha4mem : a4.mem = s.mem := by rw [ha4]; exact ha3mem
  have ha4k : a4.k = s.k := by rw [ha4]; exact ha3k
  have hrd5 : rd64 a4.mem (sp0 + 16) =
      some (sp0 + 56 + argv0.length + 1) := by
    have h := initStack_rd64_argv1 (by rw [← hstk]; exact hdisj')
    rw [← hstk, ← hsp0, hPmod] at h
    rwa [ha4mem, hmem]
  have e5 : X86Int.step stage1Im a4 =
      .ok { (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
          (some (sp0 + 56 + argv0.length + 1)) with rip := a4.rip + 1 } :=
    step_pop (by rw [ha4rip]; exact hc.d85) ha4r4 hrd5
  obtain ⟨a5, ha5⟩ : ∃ t : X86Int.St, t =
      { (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
          (some (sp0 + 56 + argv0.length + 1)) with rip := a4.rip + 1 } :=
    ⟨_, rfl⟩
  have ha5rip : a5.rip = base + 0x86 := by
    rw [ha5]; show a4.rip + 1 = base + 0x86; rw [ha4rip]
  have hlen5 : a5.regs.length = 16 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen4
  have ha5r6 : a5.reg 6 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 6 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_self (by
      show 6 < (a4.set 4 (some ((sp0 + 16 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ha5r4 : a5.reg 4 = some (sp0 + 24) := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha5mem : a5.mem = s.mem := by rw [ha5]; exact ha4mem
  have ha5k : a5.k = s.k := by rw [ha5]; exact ha4k
  have ha5r2 : a5.reg 2 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 6),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 6),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 2 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 2 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 0),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    exact hr0 2 (by decide) (by decide)
  have ha5r10 : a5.reg 10 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 6),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 6),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 10 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 10 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 0),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    exact hr0 10 (by decide) (by decide)
  have ha5r8 : a5.reg 8 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6),
      reg_set_other (by decide : (8 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6),
      reg_set_other (by decide : (8 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 8 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 8 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0),
      reg_set_other (by decide : (8 : Nat) ≠ 4)]
    exact hr0 8 (by decide) (by decide)
  have ha5r9 : a5.reg 9 = some 0 := by
    rw [ha5]
    show ({ (a4.set 4 (some ((sp0 + 16 + 8) % W))).set 6
        (some (sp0 + 56 + argv0.length + 1)) with
      rip := a4.rip + 1 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha4]
    show ({ (a3.set 4 (some ((sp0 + 8 + 8) % W))).set 6 (some (sp0 + 56)) with
      rip := a3.rip + 1 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha3]
    show ({ a2 with rip := base + 0x84 }).reg 9 = some 0
    rw [reg_with_rip, ha2]
    show ({ a1 with fl := (aluF .sub 2 (2 % W)).2, rip := a1.rip + 4 }).reg 9 = some 0
    rw [ha1]
    show ({ (s.set 4 (some ((sp0 + 8) % W))).set 0 (some 2) with
      rip := s.rip + 1 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    exact hr0 9 (by decide) (by decide)
  have reg_with_kmr : ∀ (t : X86Int.St) (k' : Kern) (m' : Mem) (x i : Nat),
      ({ t with k := k', mem := m', rip := x }).reg i = t.reg i :=
    fun _ _ _ _ _ => rfl
  -- step 6: mov eax, 2 (open)
  have e6 : X86Int.step stage1Im a5 =
      .ok { a5.set 0 (some (2 % W)) with rip := a5.rip + 5 } :=
    step_movI (by rw [ha5rip]; exact hc.d86)
  obtain ⟨a6, ha6⟩ : ∃ t : X86Int.St, t =
      { a5.set 0 (some (2 % W)) with rip := a5.rip + 5 } := ⟨_, rfl⟩
  have ha6rip : a6.rip = base + 0x8b := by
    rw [ha6]; show a5.rip + 5 = base + 0x8b; rw [ha5rip]
  have hlen6 : a6.regs.length = 16 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen5
  have ha6r0 : a6.reg 0 = some 2 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 0 = some 2
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha6r6 : a6.reg 6 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 6 =
      some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 0)]
    exact ha5r6
  have ha6r2 : a6.reg 2 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact ha5r2
  have ha6r10 : a6.reg 10 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact ha5r10
  have ha6r8 : a6.reg 8 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact ha5r8
  have ha6r9 : a6.reg 9 = some 0 := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact ha5r9
  have ha6r4 : a6.reg 4 = some (sp0 + 24) := by
    rw [ha6]
    show ({ a5.set 0 (some (2 % W)) with rip := a5.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact ha5r4
  have ha6mem : a6.mem = s.mem := by rw [ha6]; exact ha5mem
  have ha6k : a6.k = s.k := by rw [ha6]; exact ha5k
  -- step 7: mov rdi, rsi (the path)
  have e7 : X86Int.step stage1Im a6 =
      .ok { a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
        rip := a6.rip + 3 } :=
    step_mov (by rw [ha6rip]; exact hc.d8b) ha6r6
  obtain ⟨a7, ha7⟩ : ∃ t : X86Int.St, t =
      { a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
        rip := a6.rip + 3 } := ⟨_, rfl⟩
  have ha7rip : a7.rip = base + 0x8e := by
    rw [ha7]; show a6.rip + 3 = base + 0x8e; rw [ha6rip]
  have hlen7 : a7.regs.length = 16 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen6
  have ha7r7 : a7.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 7 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha7r0 : a7.reg 0 = some 2 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 0 = some 2
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact ha6r0
  have ha7r6 : a7.reg 6 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 6 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 7)]
    exact ha6r6
  have ha7r2 : a7.reg 2 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 2 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 7)]
    exact ha6r2
  have ha7r10 : a7.reg 10 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 10 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 7)]
    exact ha6r10
  have ha7r8 : a7.reg 8 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 8 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    exact ha6r8
  have ha7r9 : a7.reg 9 = some 0 := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    exact ha6r9
  have ha7r4 : a7.reg 4 = some (sp0 + 24) := by
    rw [ha7]
    show ({ a6.set 7 (some (sp0 + 56 + argv0.length + 1)) with
      rip := a6.rip + 3 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    exact ha6r4
  have ha7mem : a7.mem = s.mem := by rw [ha7]; exact ha6mem
  have ha7k : a7.k = s.k := by rw [ha7]; exact ha6k
  -- step 8: xor esi, esi
  have e8 : X86Int.step stage1Im a7 =
      .ok { a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a7.rip + 2 } :=
    step_alu_xor (by rw [ha7rip]; exact hc.d8e) ha7r6 ha7r6
  obtain ⟨a8, ha8⟩ : ∃ t : X86Int.St, t =
      { a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a7.rip + 2 } := ⟨_, rfl⟩
  have ha8rip : a8.rip = base + 0x90 := by
    rw [ha8]; show a7.rip + 2 = base + 0x90; rw [ha7rip]
  have hlen8 : a8.regs.length = 16 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).regs.length = 16
    rw [len_set]
    exact hlen7
  have ha8r6 : a8.reg 6 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 6 = some 0
    rw [reg_set_self (by
      show 6 < a7.regs.length
      omega), aluF_xor_fst_self]
  have ha8r0 : a8.reg 0 = some 2 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 0 = some 2
    rw [reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact ha7r0
  have ha8r7 : a8.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 7 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact ha7r7
  have ha8r2 : a8.reg 2 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 2 = some 0
    rw [reg_set_other (by decide : (2 : Nat) ≠ 6)]
    exact ha7r2
  have ha8r10 : a8.reg 10 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 10 = some 0
    rw [reg_set_other (by decide : (10 : Nat) ≠ 6)]
    exact ha7r10
  have ha8r8 : a8.reg 8 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 8 = some 0
    rw [reg_set_other (by decide : (8 : Nat) ≠ 6)]
    exact ha7r8
  have ha8r9 : a8.reg 9 = some 0 := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 6)]
    exact ha7r9
  have ha8r4 : a8.reg 4 = some (sp0 + 24) := by
    rw [ha8]
    show (a7.set 6 (some (aluF .xor (sp0 + 56 + argv0.length + 1) (sp0 + 56 + argv0.length + 1)).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 6)]
    exact ha7r4
  have ha8mem : a8.mem = s.mem := by rw [ha8]; exact ha7mem
  have ha8k : a8.k = s.k := by rw [ha8]; exact ha7k
  -- step 9: xor edx, edx
  have e9 : X86Int.step stage1Im a8 =
      .ok { a8.set 2 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a8.rip + 2 } :=
    step_alu_xor (by rw [ha8rip]; exact hc.d90) ha8r2 ha8r2
  obtain ⟨a9, ha9⟩ : ∃ t : X86Int.St, t =
      { a8.set 2 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a8.rip + 2 } := ⟨_, rfl⟩
  have ha9rip : a9.rip = base + 0x92 := by
    rw [ha9]; show a8.rip + 2 = base + 0x92; rw [ha8rip]
  have hlen9 : a9.regs.length = 16 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).regs.length = 16
    rw [len_set]
    exact hlen8
  have ha9r0 : a9.reg 0 = some 2 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 0 = some 2
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]
    exact ha8r0
  have ha9r7 : a9.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 7 = some (sp0 + 56 + argv0.length + 1)
    rw [reg_set_other (by decide : (7 : Nat) ≠ 2)]
    exact ha8r7
  have ha9r6 : a9.reg 6 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 6 = some 0
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact ha8r6
  have ha9r2 : a9.reg 2 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 2 = some 0
    rw [reg_set_self (by show 2 < a8.regs.length; omega), aluF_xor_fst_self]
  have ha9r10 : a9.reg 10 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 10 = some 0
    rw [reg_set_other (by decide : (10 : Nat) ≠ 2)]
    exact ha8r10
  have ha9r8 : a9.reg 8 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 8 = some 0
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    exact ha8r8
  have ha9r9 : a9.reg 9 = some 0 := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    exact ha8r9
  have ha9r4 : a9.reg 4 = some (sp0 + 24) := by
    rw [ha9]
    show (a8.set 2 (some (aluF .xor 0 0).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    exact ha8r4
  have ha9mem : a9.mem = s.mem := by rw [ha9]; exact ha8mem
  have ha9k : a9.k = s.k := by rw [ha9]; exact ha8k
  -- step 10: open(path, O_RDONLY) = 3
  have hcstr : Mem.cstr a9.mem (sp0 + 56 + argv0.length + 1) 4096 =
      some path := by
    have h := initStack_cstr_path (by rw [← hstk]; exact hdisj')
      hpath0 hpathlen
    rw [← hstk, ← hsp0] at h
    rwa [ha9mem, hmem]
  have hsc : X86Env.syscall a9.k a9.mem 2
        (sp0 + 56 + argv0.length + 1) 0 0 0 0 0 =
      some (.ret 3 ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩
        a9.mem) := by
    rw [ha9k, hk]
    simp only [X86Env.syscall]
    rw [if_neg (by simp), hcstr]
    simp [X86Env.lookup, X86Env.freeSlot, X86Env.setSlot]
  have e10 : X86Int.step stage1Im a9 =
      .ok { ((a9.set 0 (some 3)).set 1 (some (a9.rip + 2))).set 11 none with
        k := ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩,
        mem := a9.mem, rip := a9.rip + 2 } :=
    step_syscall_ret (by rw [ha9rip]; exact hc.d92) ha9r0 ha9r7 ha9r6
      ha9r2 ha9r10 ha9r8 ha9r9 hsc
  obtain ⟨a10, ha10⟩ : ∃ t : X86Int.St, t =
      { ((a9.set 0 (some 3)).set 1 (some (a9.rip + 2))).set 11 none with
        k := ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩,
        mem := a9.mem, rip := a9.rip + 2 } := ⟨_, rfl⟩
  have ha10rip : a10.rip = base + 0x94 := by
    rw [ha10]; show a9.rip + 2 = base + 0x94; rw [ha9rip]
  have hlen10 : a10.regs.length = 16 := by
    rw [ha10]
    show (((a9.set 0 (some 3)).set 1 (some (a9.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen9
  have ha10r0 : a10.reg 0 = some 3 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have ha10r7 : a10.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (7 : Nat) ≠ 11),
      reg_set_other (by decide : (7 : Nat) ≠ 1),
      reg_set_other (by decide : (7 : Nat) ≠ 0)]
    exact ha9r7
  have ha10r6 : a10.reg 6 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (6 : Nat) ≠ 11),
      reg_set_other (by decide : (6 : Nat) ≠ 1),
      reg_set_other (by decide : (6 : Nat) ≠ 0)]
    exact ha9r6
  have ha10r2 : a10.reg 2 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (2 : Nat) ≠ 11),
      reg_set_other (by decide : (2 : Nat) ≠ 1),
      reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact ha9r2
  have ha10r10 : a10.reg 10 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact ha9r10
  have ha10r8 : a10.reg 8 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact ha9r8
  have ha10r9 : a10.reg 9 = some 0 := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact ha9r9
  have ha10r4 : a10.reg 4 = some (sp0 + 24) := by
    rw [ha10, reg_with_kmr, reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact ha9r4
  have ha10mem : a10.mem = s.mem := by rw [ha10]; exact ha9mem
  have ha10k : a10.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by
    rw [ha10]
  -- step 11: test rax, rax (fd = 3, not negative)
  have e11 : X86Int.step stage1Im a10 =
      .ok { a10 with fl := (aluF .and 3 3).2, rip := a10.rip + 3 } :=
    step_test (by rw [ha10rip]; exact hc.d94) ha10r0 ha10r0
  obtain ⟨a11, ha11⟩ : ∃ t : X86Int.St, t =
      { a10 with fl := (aluF .and 3 3).2, rip := a10.rip + 3 } := ⟨_, rfl⟩
  have ha11rip : a11.rip = base + 0x97 := by
    rw [ha11]; show a10.rip + 3 = base + 0x97; rw [ha10rip]
  have ha11fl : X86Int.cond a11.fl 9 = some true := by
    rw [ha11]
    show X86Int.cond (aluF .and 3 3).2 9 = some true
    apply cond_cc9_of_sf_false
    decide
  have ha11r0 : a11.reg 0 = some 3 := by rw [ha11]; exact ha10r0
  have ha11r4 : a11.reg 4 = some (sp0 + 24) := by rw [ha11]; exact ha10r4
  have ha11mem : a11.mem = s.mem := by rw [ha11]; exact ha10mem
  have ha11k : a11.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha11]; exact ha10k
  have hlen11 : a11.regs.length = 16 := by
    rw [ha11]; show a10.regs.length = 16
    exact hlen10
  -- step 12: jns 0x9e (taken)
  have e12 : X86Int.step stage1Im a11 =
      .ok { a11 with rip := base + 0x9e } := by
    have h := step_jcc (im := stage1Im) (s := a11)
      (by rw [ha11rip]; exact hc.d97) ha11fl
    simpa using h
  obtain ⟨a12, ha12⟩ : ∃ t : X86Int.St, t =
      { a11 with rip := base + 0x9e } := ⟨_, rfl⟩
  have ha12rip : a12.rip = base + 0x9e := by rw [ha12]
  have ha12r0 : a12.reg 0 = some 3 := by
    rw [ha12]; show ({ a11 with rip := base + 0x9e }).reg 0 = some 3
    rw [reg_with_rip]; exact ha11r0
  have ha12r4 : a12.reg 4 = some (sp0 + 24) := by
    rw [ha12]; show ({ a11 with rip := base + 0x9e }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact ha11r4
  have ha12mem : a12.mem = s.mem := by rw [ha12]; exact ha11mem
  have ha12k : a12.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha12]; exact ha11k
  have hlen12 : a12.regs.length = 16 := by
    rw [ha12]; show ({ a11 with rip := base + 0x9e }).regs.length = 16
    rw [len_with_rip]; exact hlen11
  have ha12r7 : a12.reg 7 = some (sp0 + 56 + argv0.length + 1) := by rw [ha12, ha11]; exact ha10r7
  have ha12r6 : a12.reg 6 = some 0 := by rw [ha12, ha11]; exact ha10r6
  have ha12r2 : a12.reg 2 = some 0 := by rw [ha12, ha11]; exact ha10r2
  have ha12r10 : a12.reg 10 = some 0 := by rw [ha12, ha11]; exact ha10r10
  have ha12r8 : a12.reg 8 = some 0 := by rw [ha12, ha11]; exact ha10r8
  have ha12r9 : a12.reg 9 = some 0 := by rw [ha12, ha11]; exact ha10r9
  -- step 13: mov r14, rax (keep the fd)
  have e13 : X86Int.step stage1Im a12 =
      .ok { a12.set 14 (some 3) with rip := a12.rip + 3 } :=
    step_mov (by rw [ha12rip]; exact hc.d9e) ha12r0
  obtain ⟨a13, ha13⟩ : ∃ t : X86Int.St, t =
      { a12.set 14 (some 3) with rip := a12.rip + 3 } := ⟨_, rfl⟩
  have ha13rip : a13.rip = base + 0xa1 := by
    rw [ha13]; show a12.rip + 3 = base + 0xa1; rw [ha12rip]
  have hlen13 : a13.regs.length = 16 := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen12
  have ha13r14 : a13.reg 14 = some 3 := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_self (by omega)]
  have ha13r0 : a13.reg 0 = some 3 := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 0 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 14)]
    exact ha12r0
  have ha13r7 : a13.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 7 =
      some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 14)]
    exact ha12r7
  have ha13r4 : a13.reg 4 = some (sp0 + 24) := by
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 14)]
    exact ha12r4
  have ha13mem : a13.mem = s.mem := by rw [ha13]; exact ha12mem
  have ha13k : a13.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha13]; exact ha12k
  -- step 14: mov eax, 9 (mmap)
  have e14 : X86Int.step stage1Im a13 =
      .ok { a13.set 0 (some (9 % W)) with rip := a13.rip + 5 } :=
    step_movI (by rw [ha13rip]; exact hc.da1)
  obtain ⟨a14, ha14⟩ : ∃ t : X86Int.St, t =
      { a13.set 0 (some (9 % W)) with rip := a13.rip + 5 } := ⟨_, rfl⟩
  have ha14rip : a14.rip = base + 0xa6 := by
    rw [ha14]; show a13.rip + 5 = base + 0xa6; rw [ha13rip]
  have hlen14 : a14.regs.length = 16 := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen13
  have ha14r0 : a14.reg 0 = some 9 := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha14r7 : a14.reg 7 = some (sp0 + 56 + argv0.length + 1) := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 7 =
      some (sp0 + 56 + argv0.length + 1)
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 0)]
    exact ha13r7
  have ha14r14 : a14.reg 14 = some 3 := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact ha13r14
  have ha14r4 : a14.reg 4 = some (sp0 + 24) := by
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact ha13r4
  have ha14mem : a14.mem = s.mem := by rw [ha14]; exact ha13mem
  have ha14k : a14.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha14]; exact ha13k
  -- step 15: xor edi, edi
  have e15 : X86Int.step stage1Im a14 =
      .ok { a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a14.rip + 2 } :=
    step_alu_xor (by rw [ha14rip]; exact hc.da6) ha14r7 ha14r7
  obtain ⟨a15, ha15⟩ : ∃ t : X86Int.St, t =
      { a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).1) with
        fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
          (sp0 + 56 + argv0.length + 1)).2, rip := a14.rip + 2 } := ⟨_, rfl⟩
  have ha15rip : a15.rip = base + 0xa8 := by
    rw [ha15]; show a14.rip + 2 = base + 0xa8; rw [ha14rip]
  have hlen15 : a15.regs.length = 16 := by
    rw [ha15]
    show ({ a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1) with
      fl := (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).2, rip := a14.rip + 2 }).regs.length = 16
    rw [len_set]
    exact hlen14
  have ha15r7 : a15.reg 7 = some 0 := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 7 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have ha15r0 : a15.reg 0 = some 9 := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 0 = some 9
    rw [reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact ha14r0
  have ha15r14 : a15.reg 14 = some 3 := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 7)]
    exact ha14r14
  have ha15r4 : a15.reg 4 = some (sp0 + 24) := by
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 7)]
    exact ha14r4
  have ha15mem : a15.mem = s.mem := by rw [ha15]; exact ha14mem
  have ha15k : a15.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha15]; exact ha14k
  -- steps 16-18: mov esi, 65536; mov edx, 3; mov r10d, 34
  have e16 : X86Int.step stage1Im a15 =
      .ok { a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 } :=
    step_movI (by rw [ha15rip]; exact hc.da8)
  obtain ⟨a16, ha16⟩ : ∃ t : X86Int.St, t =
      { a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 } := ⟨_, rfl⟩
  have ha16rip : a16.rip = base + 0xad := by
    rw [ha16]; show a15.rip + 5 = base + 0xad; rw [ha15rip]
  have hlen16 : a16.regs.length = 16 := by
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with
      rip := a15.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen15
  have ha16r6 : a16.reg 6 = some 65536 := by
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with
      rip := a15.rip + 5 }).reg 6 = some 65536
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have e17 : X86Int.step stage1Im a16 =
      .ok { a16.set 2 (some (3 % W)) with rip := a16.rip + 5 } :=
    step_movI (by rw [ha16rip]; exact hc.dad)
  obtain ⟨a17, ha17⟩ : ∃ t : X86Int.St, t =
      { a16.set 2 (some (3 % W)) with rip := a16.rip + 5 } := ⟨_, rfl⟩
  have ha17rip : a17.rip = base + 0xb2 := by
    rw [ha17]; show a16.rip + 5 = base + 0xb2; rw [ha16rip]
  have hlen17 : a17.regs.length = 16 := by
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen16
  have e18 : X86Int.step stage1Im a17 =
      .ok { a17.set 10 (some (34 % W)) with rip := a17.rip + 6 } :=
    step_movI (by rw [ha17rip]; exact hc.db2)
  obtain ⟨a18, ha18⟩ : ∃ t : X86Int.St, t =
      { a17.set 10 (some (34 % W)) with rip := a17.rip + 6 } := ⟨_, rfl⟩
  have ha18rip : a18.rip = base + 0xb8 := by
    rw [ha18]; show a17.rip + 6 = base + 0xb8; rw [ha17rip]
  have hlen18 : a18.regs.length = 16 := by
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with
      rip := a17.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen17
  have ha18mem : a18.mem = s.mem := by rw [ha18, ha17, ha16]; exact ha15mem
  have ha18k : a18.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha18, ha17, ha16]; exact ha15k
  have ha18r4 : a18.reg 4 = some (sp0 + 24) := by
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    exact ha15r4
  -- step 19: push -1
  have hwr : ∃ stk', wr64 a18.mem (sp0 + 16) ((W - 1) % W) =
      some [⟨base, stage1Code, false⟩, stk'] ∧ stk'.base = stk.base ∧
      stk'.hi = stk.hi ∧ stk'.rw = stk.rw := by
    rw [ha18mem, hmem]
    have h := wrN_second_some (r := stk) (a := sp0 + 16)
      (bytes8 ((W - 1) % W)) hdisj' (by
      rw [hstk]
      simp [X86Env.initStack]) (by omega) (by
      have hlen8 : (bytes8 ((W - 1) % W)).length = 8 := by simp [bytes8]
      rw [hlen8]
      have hge : X86Env.stackRoom + 24 ≤ stk.data.length := by
        rw [hdatalen]
        have := pages_ge (56 + ((argv0.length + 1) + (path.length + 1)) + 1)
        omega
      unfold Region.hi
      omega)
    obtain ⟨stk', h1, h2, h3, h4⟩ := h
    exact ⟨stk', h1, h2, h3, h4⟩
  obtain ⟨stk', hwr, hstk'b, hstk'hi, hstk'rw⟩ := hwr
  have hwshape : wshape [⟨base, stage1Code, false⟩, stk'] =
      wshape a18.mem := by
    have h := wrN_wshape (bytes8 ((W - 1) % W)) a18.mem (sp0 + 16) hwr
    exact h
  have e19 : X86Int.step stage1Im a18 =
      .ok { a18.set 4 (some ((sp0 + 24 + W - 8) % W)) with
        mem := [⟨base, stage1Code, false⟩, stk'], rip := a18.rip + 2 } :=
    step_pushI (by rw [ha18rip]; exact hc.db8) ha18r4 (by
      rw [show (sp0 + 24 + W - 8) % W = sp0 + 16 from by
        rw [show sp0 + 24 + W - 8 = (sp0 + 16) + W from by omega,
          Nat.add_mod_right]
        exact Nat.mod_eq_of_lt (by omega)]
      exact hwr)
  obtain ⟨a19, ha19⟩ : ∃ t : X86Int.St, t =
      { a18.set 4 (some ((sp0 + 24 + W - 8) % W)) with
        mem := [⟨base, stage1Code, false⟩, stk'], rip := a18.rip + 2 } :=
    ⟨_, rfl⟩
  have ha19rip : a19.rip = base + 0xba := by
    rw [ha19]; show a18.rip + 2 = base + 0xba; rw [ha18rip]
  have hlen19 : a19.regs.length = 16 := by
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).regs.length = 16
    rw [len_set]
    exact hlen18
  have ha19r4 : a19.reg 4 = some (sp0 + 16) := by
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 4 = some (sp0 + 16)
    rw [reg_set_self (by omega),
      show (sp0 + 24 + W - 8) % W = sp0 + 16 from by
        rw [show sp0 + 24 + W - 8 = (sp0 + 16) + W from by omega,
          Nat.add_mod_right]
        exact Nat.mod_eq_of_lt (by omega)]
  have ha19mem : a19.mem = [⟨base, stage1Code, false⟩, stk'] := by
    rw [ha19]
  have ha19k : a19.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha19]; exact ha18k
  -- step 20: pop r8 (the -1 just pushed)
  have hrd20 : rd64 a19.mem (sp0 + 16) = some (W - 1) := by
    rw [ha19mem]
    have h := rd64_wr64 hwr
    rw [Nat.mod_mod, Nat.mod_eq_of_lt (by unfold W; omega)] at h
    exact h
  have e20 : X86Int.step stage1Im a19 =
      .ok { (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := a19.rip + 2 } :=
    step_pop (by rw [ha19rip]; exact hc.dba) ha19r4 hrd20
  obtain ⟨a20, ha20⟩ : ∃ t : X86Int.St, t =
      { (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := a19.rip + 2 } := ⟨_, rfl⟩
  have ha20rip : a20.rip = base + 0xbc := by
    rw [ha20]; show a19.rip + 2 = base + 0xbc; rw [ha19rip]
  have hlen20 : a20.regs.length = 16 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]
    exact hlen19
  have ha20r9 : a20.reg 9 = some 0 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 8),
      reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 9 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [ha14]
    show ({ a13.set 0 (some (9 % W)) with rip := a13.rip + 5 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    rw [ha13]
    show ({ a12.set 14 (some 3) with rip := a12.rip + 3 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 14)]
    exact ha12r9
  have ha20mem : a20.mem = [⟨base, stage1Code, false⟩, stk'] := by rw [ha20]; exact ha19mem
  have ha20k : a20.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha20]; exact ha19k
  -- step 21: xor r9d, r9d
  have e21 : X86Int.step stage1Im a20 =
      .ok { a20.set 9 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a20.rip + 3 } :=
    step_alu_xor (by rw [ha20rip]; exact hc.dbc) ha20r9 ha20r9
  obtain ⟨a21, ha21⟩ : ∃ t : X86Int.St, t =
      { a20.set 9 (some (aluF .xor 0 0).1) with
        fl := (aluF .xor 0 0).2, rip := a20.rip + 3 } := ⟨_, rfl⟩
  have ha21rip : a21.rip = base + 0xbf := by
    rw [ha21]; show a20.rip + 3 = base + 0xbf; rw [ha20rip]
  have hlen21 : a21.regs.length = 16 := by
    rw [ha21]
    show ({ a20.set 9 (some (aluF .xor 0 0).1) with
      fl := (aluF .xor 0 0).2, rip := a20.rip + 3 }).regs.length = 16
    rw [len_set]
    exact hlen20
  -- register conclusions at a21, by unwinding the set-chains
  have ha21r9 : a21.reg 9 = some 0 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 9 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have ha20r8 : a20.reg 8 = some (W - 1) := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 8 = some (W - 1)
    rw [reg_with_rip, reg_set_self (by
      show 8 < (a19.set 4 (some ((sp0 + 16 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ha21r8 : a21.reg 8 = some (W - 1) := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 8 = some (W - 1)
    rw [reg_set_other (by decide : (8 : Nat) ≠ 9)]
    exact ha20r8
  have ha20r4 : a20.reg 4 = some (sp0 + 24) := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 8),
      reg_set_self (by omega), Nat.mod_eq_of_lt (by omega)]
  have ha21r4 : a21.reg 4 = some (sp0 + 24) := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 9)]
    exact ha20r4
  have ha20r0 : a20.reg 0 = some 9 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 8),
      reg_set_other (by decide : (0 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 0 = some 9
    rw [reg_set_other (by decide : (0 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact ha15r0
  have ha21r0 : a21.reg 0 = some 9 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 0 = some 9
    rw [reg_set_other (by decide : (0 : Nat) ≠ 9)]
    exact ha20r0
  have ha20r7 : a20.reg 7 = some 0 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 8),
      reg_set_other (by decide : (7 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 7 = some 0
    rw [reg_set_other (by decide : (7 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 7 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact ha15r7
  have ha21r7 : a21.reg 7 = some 0 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 7 = some 0
    rw [reg_set_other (by decide : (7 : Nat) ≠ 9)]
    exact ha20r7
  have ha20r6 : a20.reg 6 = some 65536 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 6 = some 65536
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 8),
      reg_set_other (by decide : (6 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 6 = some 65536
    rw [reg_set_other (by decide : (6 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 6 =
      some 65536
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 6 =
      some 65536
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact ha16r6
  have ha21r6 : a21.reg 6 = some 65536 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 6 = some 65536
    rw [reg_set_other (by decide : (6 : Nat) ≠ 9)]
    exact ha20r6
  have ha20r2 : a20.reg 2 = some 3 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 2 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 8),
      reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 2 = some 3
    rw [reg_set_other (by decide : (2 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 2 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 2 = some 3
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha21r2 : a21.reg 2 = some 3 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 2 = some 3
    rw [reg_set_other (by decide : (2 : Nat) ≠ 9)]
    exact ha20r2
  have ha20r10 : a20.reg 10 = some 34 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 10 = some 34
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 8),
      reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have ha21r10 : a21.reg 10 = some 34 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 9)]
    exact ha20r10
  have ha20r14 : a20.reg 14 = some 3 := by
    rw [ha20]
    show ({ (a19.set 4 (some ((sp0 + 16 + 8) % W))).set 8 (some (W - 1)) with
      rip := a19.rip + 2 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 8),
      reg_set_other (by decide : (14 : Nat) ≠ 4)]
    rw [ha19]
    show (a18.set 4 (some ((sp0 + 24 + W - 8) % W))).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 4)]
    rw [ha18]
    show ({ a17.set 10 (some (34 % W)) with rip := a17.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 10)]
    rw [ha17]
    show ({ a16.set 2 (some (3 % W)) with rip := a16.rip + 5 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [ha16]
    show ({ a15.set 6 (some (65536 % W)) with rip := a15.rip + 5 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [ha15]
    show (a14.set 7 (some (aluF .xor (sp0 + 56 + argv0.length + 1)
        (sp0 + 56 + argv0.length + 1)).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 7)]
    exact ha14r14
  have ha21r14 : a21.reg 14 = some 3 := by
    rw [ha21]
    show (a20.set 9 (some (aluF .xor 0 0).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 9)]
    exact ha20r14
  have ha21mem : a21.mem = [⟨base, stage1Code, false⟩, stk'] := by rw [ha21]; exact ha20mem
  have ha21k : a21.k =
      ⟨[(path, f)], [some ⟨f, 0⟩], stdinL, [], oracle⟩ := by rw [ha21]; exact ha20k
  -- fold the run
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok a1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ha1, run_zero]
  have r2 : run stage1Im 1 (Out.ok a1) = Out.ok a2 := by
    rw [run_succ_ok stage1Im 0 a1, e2, ← ha2, run_zero]
  have r3 : run stage1Im 1 (Out.ok a2) = Out.ok a3 := by
    rw [run_succ_ok stage1Im 0 a2, e3, ← ha3, run_zero]
  have r4 : run stage1Im 1 (Out.ok a3) = Out.ok a4 := by
    rw [run_succ_ok stage1Im 0 a3, e4, ← ha4, run_zero]
  have r5 : run stage1Im 1 (Out.ok a4) = Out.ok a5 := by
    rw [run_succ_ok stage1Im 0 a4, e5, ← ha5, run_zero]
  have r6 : run stage1Im 1 (Out.ok a5) = Out.ok a6 := by
    rw [run_succ_ok stage1Im 0 a5, e6, ← ha6, run_zero]
  have r7 : run stage1Im 1 (Out.ok a6) = Out.ok a7 := by
    rw [run_succ_ok stage1Im 0 a6, e7, ← ha7, run_zero]
  have r8 : run stage1Im 1 (Out.ok a7) = Out.ok a8 := by
    rw [run_succ_ok stage1Im 0 a7, e8, ← ha8, run_zero]
  have r9 : run stage1Im 1 (Out.ok a8) = Out.ok a9 := by
    rw [run_succ_ok stage1Im 0 a8, e9, ← ha9, run_zero]
  have r10 : run stage1Im 1 (Out.ok a9) = Out.ok a10 := by
    rw [run_succ_ok stage1Im 0 a9, e10, ← ha10, run_zero]
  have r11 : run stage1Im 1 (Out.ok a10) = Out.ok a11 := by
    rw [run_succ_ok stage1Im 0 a10, e11, ← ha11, run_zero]
  have r12 : run stage1Im 1 (Out.ok a11) = Out.ok a12 := by
    rw [run_succ_ok stage1Im 0 a11, e12, ← ha12, run_zero]
  have r13 : run stage1Im 1 (Out.ok a12) = Out.ok a13 := by
    rw [run_succ_ok stage1Im 0 a12, e13, ← ha13, run_zero]
  have r14 : run stage1Im 1 (Out.ok a13) = Out.ok a14 := by
    rw [run_succ_ok stage1Im 0 a13, e14, ← ha14, run_zero]
  have r15 : run stage1Im 1 (Out.ok a14) = Out.ok a15 := by
    rw [run_succ_ok stage1Im 0 a14, e15, ← ha15, run_zero]
  have r16 : run stage1Im 1 (Out.ok a15) = Out.ok a16 := by
    rw [run_succ_ok stage1Im 0 a15, e16, ← ha16, run_zero]
  have r17 : run stage1Im 1 (Out.ok a16) = Out.ok a17 := by
    rw [run_succ_ok stage1Im 0 a16, e17, ← ha17, run_zero]
  have r18 : run stage1Im 1 (Out.ok a17) = Out.ok a18 := by
    rw [run_succ_ok stage1Im 0 a17, e18, ← ha18, run_zero]
  have r19 : run stage1Im 1 (Out.ok a18) = Out.ok a19 := by
    rw [run_succ_ok stage1Im 0 a18, e19, ← ha19, run_zero]
  have r20 : run stage1Im 1 (Out.ok a19) = Out.ok a20 := by
    rw [run_succ_ok stage1Im 0 a19, e20, ← ha20, run_zero]
  have r21 : run stage1Im 1 (Out.ok a20) = Out.ok a21 := by
    rw [run_succ_ok stage1Im 0 a20, e21, ← ha21, run_zero]
  have hrun : run stage1Im 21 (Out.ok s) = Out.ok a21 := by
    rw [show (21 : Nat) = 1 + 20 from rfl, run_add stage1Im 1 20, r1,
      show (20 : Nat) = 1 + 19 from rfl, run_add stage1Im 1 19, r2,
      show (19 : Nat) = 1 + 18 from rfl, run_add stage1Im 1 18, r3,
      show (18 : Nat) = 1 + 17 from rfl, run_add stage1Im 1 17, r4,
      show (17 : Nat) = 1 + 16 from rfl, run_add stage1Im 1 16, r5,
      show (16 : Nat) = 1 + 15 from rfl, run_add stage1Im 1 15, r6,
      show (15 : Nat) = 1 + 14 from rfl, run_add stage1Im 1 14, r7,
      show (14 : Nat) = 1 + 13 from rfl, run_add stage1Im 1 13, r8,
      show (13 : Nat) = 1 + 12 from rfl, run_add stage1Im 1 12, r9,
      show (12 : Nat) = 1 + 11 from rfl, run_add stage1Im 1 11, r10,
      show (11 : Nat) = 1 + 10 from rfl, run_add stage1Im 1 10, r11,
      show (10 : Nat) = 1 + 9 from rfl, run_add stage1Im 1 9, r12,
      show (9 : Nat) = 1 + 8 from rfl, run_add stage1Im 1 8, r13,
      show (8 : Nat) = 1 + 7 from rfl, run_add stage1Im 1 7, r14,
      show (7 : Nat) = 1 + 6 from rfl, run_add stage1Im 1 6, r15,
      show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r16,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r17,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r18,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r19,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r20, r21]
  exact ⟨a21, hrun, ha21rip, ha21r0, ha21r7, ha21r6, ha21r2, ha21r10,
    ha21r8, ha21r9, ha21r14, ha21r4, ha21k, by
      rw [ha21mem, hwshape, ha18mem], hlen21⟩

/-- The initial `mmap` is legal for a placement in the room
interval, and six further steps reach the read loop head with the
file buffer mapped at `A`, capacity 65536 in `r12`, count 0 in
`r13`, one oracle entry consumed. -/
theorem loader_mmap1_to_readloop {f : List Nat} {A sp0 : Nat}
    {s s' : X86Int.St}
    (hrip : s'.rip = base + 0xbf) (hlen16 : s'.regs.length = 16)
    (hr0 : s'.reg 0 = some 9) (hr7 : s'.reg 7 = some 0)
    (hr6 : s'.reg 6 = some 65536) (hr2 : s'.reg 2 = some 3)
    (hr10 : s'.reg 10 = some 34) (hr8 : s'.reg 8 = some (W - 1))
    (hr9 : s'.reg 9 = some 0) (hr14 : s'.reg 14 = some 3)
    (hr4 : s'.reg 4 = some (sp0 + 24))
    (hr13 : ∃ v, s'.reg 13 = some v)
    (hshape : wshape s'.mem = wshape s.mem)
    (hhead : s'.k.oracle.head? = some (.at A))
    (hAal : A % page = 0) (hApos : 0 < A)
    (hbound : A + stage1Mf f ≤ 2 ^ 47)
    (hfree : s.mem.free A (stage1Mf f) = true) :
    OracleLegalAt s' ∧
    ∃ s'', run stage1Im 6 (Out.ok s') = Out.ok s'' ∧
      s''.rip = base + 0xd7 ∧
      s''.reg 3 = some A ∧ s''.reg 12 = some 65536 ∧
      s''.reg 13 = some 0 ∧ s''.reg 14 = some 3 ∧
      s''.reg 4 = some (sp0 + 24) ∧
      (∃ v, s''.reg 0 = some v) ∧ (∃ v, s''.reg 10 = some v) ∧
      (∃ v, s''.reg 8 = some v) ∧ (∃ v, s''.reg 9 = some v) ∧
      s''.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] ∧
      s''.k = (Kern.next s'.k).2 ∧ s''.regs.length = 16 := by
  have hc := stage1_loaderCode
  have hpages : X86Env.pages 65536 = 65536 := by decide
  have hMf : 65536 ≤ stage1Mf f := by
    unfold stage1Mf arenaFileBytes
    have h := pages_ge (65536 + 2 * f.length)
    omega
  have hfree' : s'.mem.free A 65536 = true := by
    have h1 : s'.mem.free A (stage1Mf f) = true := by
      rw [free_of_wshape hshape]
      exact hfree
    have h2 : s'.mem.free A (65536 + (stage1Mf f - 65536)) = true := by
      rw [Nat.add_sub_cancel' hMf]
      exact h1
    exact free_prefix h2
  have hplace : X86Env.place s'.mem (some (.at A)) (X86Env.pages 65536) =
      some (some A) := by
    rw [hpages]
    exact place_at hAal hApos (by omega) hfree'
  have hlegal : OracleLegalAt s' :=
    oracleLegalAt_mmap (by rw [hrip]; exact hc.dbf) hr0 hr7 hr6 hr2 hr10
      hr8 hr9 (by decide) hhead hplace
  have hsc : X86Env.syscall s'.k s'.mem 9 0 65536 PROT_RW
        MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret A (Kern.next s'.k).2
        (s'.mem ++ [⟨A, List.replicate 65536 0, true⟩])) := by
    have hnext1 : (Kern.next s'.k).1 = some (.at A) :=
      kern_next_of_head hhead
    simp only [X86Env.syscall]
    rw [if_neg (by simp [PROT_RW, MAP_PRIVATE_ANON]),
      if_neg (by decide : ¬ ((65536 : Nat) = 0))]
    cases hnext : Kern.next s'.k with
    | mk c k1 =>
      rw [hnext] at hnext1
      simp only [] at hnext1
      subst hnext1
      simp only [hplace]
      rw [hpages]
      rfl
  have e1 : X86Int.step stage1Im s' =
      .ok { ((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11 none with
        k := (Kern.next s'.k).2,
        mem := s'.mem ++ [⟨A, List.replicate 65536 0, true⟩],
        rip := s'.rip + 2 } :=
    step_syscall_ret (by rw [hrip]; exact hc.dbf) hr0 hr7 hr6 hr2 hr10
      hr8 hr9 hsc
  obtain ⟨b1, hb1⟩ : ∃ t : X86Int.St, t =
      { ((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11 none with
        k := (Kern.next s'.k).2,
        mem := s'.mem ++ [⟨A, List.replicate 65536 0, true⟩],
        rip := s'.rip + 2 } := ⟨_, rfl⟩
  have hb1rip : b1.rip = base + 0xc1 := by
    rw [hb1]; show s'.rip + 2 = base + 0xc1; rw [hrip]
  have hlenb1 : b1.regs.length = 16 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen16
  have hb1r0 : b1.reg 0 = some A := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 0 = some A
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hb1r4 : b1.reg 4 = some (sp0 + 24) := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4
  have hb1r14 : b1.reg 14 = some 3 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 11),
      reg_set_other (by decide : (14 : Nat) ≠ 1),
      reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hr14
  obtain ⟨v13, hb1r13⟩ : ∃ v, b1.reg 13 = some v := by
    obtain ⟨v, hv⟩ := hr13
    exact ⟨v, by
      rw [hb1]
      show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
        none).reg 13 = some v
      rw [reg_set_other (by decide : (13 : Nat) ≠ 11),
        reg_set_other (by decide : (13 : Nat) ≠ 1),
        reg_set_other (by decide : (13 : Nat) ≠ 0)]
      exact hv⟩
  have hb1mem : b1.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by
    rw [hb1]
  have hb1k : b1.k = (Kern.next s'.k).2 := by rw [hb1]
  -- test rax, rax: A < 2^63 so not negative
  have e2 : X86Int.step stage1Im b1 =
      .ok { b1 with fl := (aluF .and A A).2, rip := b1.rip + 3 } :=
    step_test (by rw [hb1rip]; exact hc.dc1) hb1r0 hb1r0
  obtain ⟨b2, hb2⟩ : ∃ t : X86Int.St, t =
      { b1 with fl := (aluF .and A A).2, rip := b1.rip + 3 } := ⟨_, rfl⟩
  have hb2rip : b2.rip = base + 0xc4 := by
    rw [hb2]; show b1.rip + 3 = base + 0xc4; rw [hb1rip]
  have hb2fl : X86Int.cond b2.fl 9 = some true := by
    rw [hb2]
    show X86Int.cond (aluF .and A A).2 9 = some true
    apply cond_cc9_of_sf_false
    have hland : Nat.land A A = A := by simp
    have hmsb : X86Int.msb A = false := by
      simp only [X86Int.msb]
      rw [Nat.div_eq_of_lt (by omega : A < 2 ^ 63)]
      decide
    simp [aluF, hland, hmsb]
  have hb2r0 : b2.reg 0 = some A := by rw [hb2]; exact hb1r0
  have hb2r4 : b2.reg 4 = some (sp0 + 24) := by rw [hb2]; exact hb1r4
  have hb2r14 : b2.reg 14 = some 3 := by rw [hb2]; exact hb1r14
  have hb2r13 : b2.reg 13 = some v13 := by rw [hb2]; exact hb1r13
  have hb2mem : b2.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb2]; exact hb1mem
  have hb2k : b2.k = (Kern.next s'.k).2 := by rw [hb2]; exact hb1k
  have hlenb2 : b2.regs.length = 16 := by
    rw [hb2]; show b1.regs.length = 16
    exact hlenb1
  -- jns 0xcb (taken)
  have e3 : X86Int.step stage1Im b2 = .ok { b2 with rip := base + 0xcb } := by
    have h := step_jcc (im := stage1Im) (s := b2)
      (by rw [hb2rip]; exact hc.dc4) hb2fl
    simpa using h
  obtain ⟨b3, hb3⟩ : ∃ t : X86Int.St, t =
      { b2 with rip := base + 0xcb } := ⟨_, rfl⟩
  have hb3rip : b3.rip = base + 0xcb := by rw [hb3]
  have hb3r0 : b3.reg 0 = some A := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 0 = some A
    rw [reg_with_rip]; exact hb2r0
  have hb3r4 : b3.reg 4 = some (sp0 + 24) := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hb2r4
  have hb3r14 : b3.reg 14 = some 3 := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 14 = some 3
    rw [reg_with_rip]; exact hb2r14
  have hb3r13 : b3.reg 13 = some v13 := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).reg 13 = some v13
    rw [reg_with_rip]; exact hb2r13
  have hb3mem : b3.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb3]; exact hb2mem
  have hb3k : b3.k = (Kern.next s'.k).2 := by rw [hb3]; exact hb2k
  have hlenb3 : b3.regs.length = 16 := by
    rw [hb3]; show ({ b2 with rip := base + 0xcb }).regs.length = 16
    rw [len_with_rip]; exact hlenb2
  -- mov rbx, rax
  have e4 : X86Int.step stage1Im b3 =
      .ok { b3.set 3 (some A) with rip := b3.rip + 3 } :=
    step_mov (by rw [hb3rip]; exact hc.dcb) hb3r0
  obtain ⟨b4, hb4⟩ : ∃ t : X86Int.St, t =
      { b3.set 3 (some A) with rip := b3.rip + 3 } := ⟨_, rfl⟩
  have hb4rip : b4.rip = base + 0xce := by
    rw [hb4]; show b3.rip + 3 = base + 0xce; rw [hb3rip]
  have hlenb4 : b4.regs.length = 16 := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlenb3
  have hb4r3 : b4.reg 3 = some A := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hb4r4 : b4.reg 4 = some (sp0 + 24) := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 3)]
    exact hb3r4
  have hb4r14 : b4.reg 14 = some 3 := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 3)]
    exact hb3r14
  have hb4r13 : b4.reg 13 = some v13 := by
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 13 = some v13
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 3)]
    exact hb3r13
  have hb4mem : b4.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb4]; exact hb3mem
  have hb4k : b4.k = (Kern.next s'.k).2 := by rw [hb4]; exact hb3k
  -- mov r12d, 65536
  have e5 : X86Int.step stage1Im b4 =
      .ok { b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 } :=
    step_movI (by rw [hb4rip]; exact hc.dce)
  obtain ⟨b5, hb5⟩ : ∃ t : X86Int.St, t =
      { b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 } := ⟨_, rfl⟩
  have hb5rip : b5.rip = base + 0xd4 := by
    rw [hb5]; show b4.rip + 6 = base + 0xd4; rw [hb4rip]
  have hlenb5 : b5.regs.length = 16 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlenb4
  have hb5r12 : b5.reg 12 = some 65536 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 12 = some 65536
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hb5r3 : b5.reg 3 = some A := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 12)]
    exact hb4r3
  have hb5r4 : b5.reg 4 = some (sp0 + 24) := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 12)]
    exact hb4r4
  have hb5r14 : b5.reg 14 = some 3 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 12)]
    exact hb4r14
  have hb5r13 : b5.reg 13 = some v13 := by
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with rip := b4.rip + 6 }).reg 13 =
      some v13
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 12)]
    exact hb4r13
  have hb5mem : b5.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb5]; exact hb4mem
  have hb5k : b5.k = (Kern.next s'.k).2 := by rw [hb5]; exact hb4k
  -- xor r13, r13
  have e6 : X86Int.step stage1Im b5 =
      .ok { b5.set 13 (some (aluF .xor v13 v13).1) with
        fl := (aluF .xor v13 v13).2, rip := b5.rip + 3 } :=
    step_alu_xor (by rw [hb5rip]; exact hc.dd4) hb5r13 hb5r13
  obtain ⟨b6, hb6⟩ : ∃ t : X86Int.St, t =
      { b5.set 13 (some (aluF .xor v13 v13).1) with
        fl := (aluF .xor v13 v13).2, rip := b5.rip + 3 } := ⟨_, rfl⟩
  have hb6rip : b6.rip = base + 0xd7 := by
    rw [hb6]; show b5.rip + 3 = base + 0xd7; rw [hb5rip]
  have hlenb6 : b6.regs.length = 16 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).regs.length = 16
    rw [len_set]
    exact hlenb5
  have hb6r13 : b6.reg 13 = some 0 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 13 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hb6r3 : b6.reg 3 = some A := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 13)]
    exact hb5r3
  have hb6r12 : b6.reg 12 = some 65536 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 12 = some 65536
    rw [reg_set_other (by decide : (12 : Nat) ≠ 13)]
    exact hb5r12
  have hb6r14 : b6.reg 14 = some 3 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 13)]
    exact hb5r14
  have hb6r4 : b6.reg 4 = some (sp0 + 24) := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 13)]
    exact hb5r4
  have hb6mem : b6.mem = s'.mem ++ [⟨A, List.replicate 65536 0, true⟩] := by rw [hb6]; exact hb5mem
  have hb6k : b6.k = (Kern.next s'.k).2 := by rw [hb6]; exact hb5k
  have hb6r0 : b6.reg 0 = some A := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 0 = some A
    rw [reg_set_other (by decide : (0 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 0 = some A
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 0 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 0 = some A
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r0
  have hb1r10 : b1.reg 10 = some 34 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact hr10
  have hb6r10 : b6.reg 10 = some 34 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 10 = some 34
    rw [reg_set_other (by decide : (10 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 10 = some 34
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 10 = some 34
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r10
  have hb1r8 : b1.reg 8 = some (W - 1) := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 8 = some (W - 1)
    rw [reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hr8
  have hb6r8 : b6.reg 8 = some (W - 1) := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 8 = some (W - 1)
    rw [reg_set_other (by decide : (8 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 8 = some (W - 1)
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 8 =
      some (W - 1)
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 8 = some (W - 1)
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r8
  have hb1r9 : b1.reg 9 = some 0 := by
    rw [hb1]
    show (((s'.set 0 (some A)).set 1 (some (s'.rip + 2))).set 11
      none).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hr9
  have hb6r9 : b6.reg 9 = some 0 := by
    rw [hb6]
    show (b5.set 13 (some (aluF .xor v13 v13).1)).reg 9 = some 0
    rw [reg_set_other (by decide : (9 : Nat) ≠ 13)]
    rw [hb5]
    show ({ b4.set 12 (some (65536 % W)) with
      rip := b4.rip + 6 }).reg 9 = some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 12)]
    rw [hb4]
    show ({ b3.set 3 (some A) with rip := b3.rip + 3 }).reg 9 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 3)]
    rw [hb3]
    show ({ b2 with rip := base + 0xcb }).reg 9 = some 0
    rw [reg_with_rip]
    rw [hb2]
    exact hb1r9
  have r1 : run stage1Im 1 (Out.ok s') = Out.ok b1 := by
    rw [run_succ_ok stage1Im 0 s', e1, ← hb1, run_zero]
  have r2 : run stage1Im 1 (Out.ok b1) = Out.ok b2 := by
    rw [run_succ_ok stage1Im 0 b1, e2, ← hb2, run_zero]
  have r3 : run stage1Im 1 (Out.ok b2) = Out.ok b3 := by
    rw [run_succ_ok stage1Im 0 b2, e3, ← hb3, run_zero]
  have r4 : run stage1Im 1 (Out.ok b3) = Out.ok b4 := by
    rw [run_succ_ok stage1Im 0 b3, e4, ← hb4, run_zero]
  have r5 : run stage1Im 1 (Out.ok b4) = Out.ok b5 := by
    rw [run_succ_ok stage1Im 0 b4, e5, ← hb5, run_zero]
  have r6 : run stage1Im 1 (Out.ok b5) = Out.ok b6 := by
    rw [run_succ_ok stage1Im 0 b5, e6, ← hb6, run_zero]
  have hrun : run stage1Im 6 (Out.ok s') = Out.ok b6 := by
    rw [show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r5, r6]
  exact ⟨hlegal, b6, hrun, hb6rip, hb6r3, hb6r12, hb6r13, hb6r14, hb6r4,
    ⟨A, hb6r0⟩, ⟨34, hb6r10⟩, ⟨W - 1, hb6r8⟩, ⟨0, hb6r9⟩,
    hb6mem, hb6k, hlenb6⟩

/-! ### Read-loop infrastructure -/

theorem step_mov32 {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {d r len v : Nat} (hi : im s.rip = some (.mov32 d r, len))
    (hr : s.reg r = some v) :
    X86Int.step im s =
      .ok { s.set d (some (v % 2 ^ 32)) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hr, lift, bind, Option.bind, pure]

theorem step_st {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {b disp r len x v : Nat} {m : Mem}
    (hi : im s.rip = some (.st b disp r, len)) (hx : s.reg b = some x)
    (hv : s.reg r = some v) (hm : wr64 s.mem (ea x disp) v = some m) :
    X86Int.step im s =
      .ok { s with mem := m, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hv, hm, lift, bind, Option.bind, pure]

theorem step_st8 {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {b disp r len x v : Nat} {m : Mem}
    (hi : im s.rip = some (.st8 b disp r, len)) (hx : s.reg b = some x)
    (hv : s.reg r = some v) (hm : s.mem.wr (ea x disp) (v % 256) = some m) :
    X86Int.step im s =
      .ok { s with mem := m, rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hx, hv, hm, lift, bind, Option.bind, pure]

theorem pages_mono {a b : Nat} (h : a ≤ b) :
    X86Env.pages a ≤ X86Env.pages b := by
  unfold X86Env.pages
  have hdiv : (a + page - 1) / page ≤ (b + page - 1) / page :=
    Nat.div_le_div_right (by omega)
  have hpage : 0 < page := by decide
  exact Nat.mul_le_mul_right page hdiv

theorem mem_wr_third {c1 c2 r : Region} {a v : Nat}
    (hdisj1 : c1.hi ≤ r.base ∨ r.hi ≤ c1.base)
    (hdisj2 : c2.hi ≤ r.base ∨ r.hi ≤ c2.base)
    (hge : r.base ≤ a) (hlt : a < r.hi) (hrw : r.rw = true) :
    Mem.wr [c1, c2, r] a v =
      some [c1, c2, { r with data := r.data.set (a - r.base) (v % 256) }] := by
  have hnot1 : ¬ (c1.base ≤ a ∧ a < c1.hi) := by
    rcases hdisj1 with h | h <;> omega
  have hnot2 : ¬ (c2.base ≤ a ∧ a < c2.hi) := by
    rcases hdisj2 with h | h <;> omega
  simp [Mem.wr, Mem.find, hnot1, hnot2, hge, hlt, hrw]

theorem wrN_third_some : ∀ (bs : List Nat) {c1 c2 r : Region} {a : Nat},
    (c1.hi ≤ r.base ∨ r.hi ≤ c1.base) →
    (c2.hi ≤ r.base ∨ r.hi ≤ c2.base) → r.rw = true → r.base ≤ a →
    a + bs.length ≤ r.hi →
    ∃ r', Mem.wrN [c1, c2, r] a bs = some [c1, c2, r'] ∧
      r'.base = r.base ∧ r'.hi = r.hi ∧ r'.rw = r.rw
  | [], c1, c2, r, a, _, _, _, _, _ =>
    ⟨r, by simp [Mem.wrN], rfl, rfl, rfl⟩
  | b :: t, c1, c2, r, a, hdisj1, hdisj2, hrw, hge, hhi => by
    have hlt : a < r.hi := by
      simp at hhi
      omega
    rw [Mem.wrN, mem_wr_third hdisj1 hdisj2 hge hlt hrw]
    have hlen : ({ r with data := r.data.set (a - r.base) (b % 256) } :
        Region).hi = r.hi := by
      simp [Region.hi, List.length_set]
    have hd1 : c1.hi ≤ r.base ∨
        ({ r with data := r.data.set (a - r.base) (b % 256) } :
          Region).hi ≤ c1.base := by
      rcases hdisj1 with h | h
      · exact Or.inl h
      · exact Or.inr (by rw [hlen]; exact h)
    have hd2 : c2.hi ≤ r.base ∨
        ({ r with data := r.data.set (a - r.base) (b % 256) } :
          Region).hi ≤ c2.base := by
      rcases hdisj2 with h | h
      · exact Or.inl h
      · exact Or.inr (by rw [hlen]; exact h)
    obtain ⟨r'', hr'', hb, hh, hrw'⟩ :=
      wrN_third_some (c1 := c1) (c2 := c2)
        (r := { r with data := r.data.set (a - r.base) (b % 256) })
        (a := a + 1) t hd1 hd2 (by simpa using hrw)
        (by show r.base ≤ a + 1; omega) (by
        rw [hlen]
        simp at hhi
        omega)
    exact ⟨r'', by simpa using hr'', by simpa using hb,
      by rw [hh, hlen], by simpa using hrw'⟩

theorem rdN_combine {m : Mem} :
    ∀ {a n : Nat} {xs : List Nat}, xs.length = n →
      m.rdN a n = some xs →
      ∀ {k : Nat} {ys : List Nat}, ys.length = k →
        m.rdN (a + n) k = some ys →
        m.rdN a (n + k) = some (xs ++ ys) := by
  intro a n
  induction n generalizing a with
  | zero =>
    intro xs hxs h k ys hys h2
    have : xs = [] := by cases xs <;> simp_all
    subst this
    simpa using h2
  | succ j ih =>
    intro xs hxs h k ys hys h2
    cases xs with
    | nil => simp at hxs
    | cons b t =>
      have ht : t.length = j := by simpa using hxs
      obtain ⟨hb, ht2⟩ := rdN_cons h
      have hrec : m.rdN (a + 1) (j + k) = some (t ++ ys) := by
        apply ih ht ht2 hys
        rw [show a + 1 + j = a + (j + 1) from by omega]
        exact h2
      rw [show j + 1 + k = (j + k) + 1 from by omega]
      simp only [Mem.rdN]
      simp [hb, hrec]

theorem rdN_wrN_prefix : ∀ (bs : List Nat) {m m' : Mem} {a k : Nat},
    m.wrN (a + k) bs = some m' → m'.rdN a k = m.rdN a k
  | bs, m, m', a, 0, _ => rfl
  | bs, m, m', a, k + 1, h => by
    have h1 : m'.rd a = m.rd a :=
      rd_wrN_other bs m (a + (k + 1)) a h (fun i _ => by omega)
    have h2 : m'.rdN (a + 1) k = m.rdN (a + 1) k := by
      have h' : m.wrN (a + 1 + k) bs = some m' := by
        rwa [show a + 1 + k = a + (k + 1) from by omega]
      exact rdN_wrN_prefix bs h'
    simp only [Mem.rdN, h1, h2]

theorem syscall_read_file {k : Kern} {m m' : Mem} {f : List Nat}
    {buf count want A : Nat} (hfd : k.fds = [some ⟨f, count⟩])
    (hhead : k.oracle.head? = some (.at A))
    (hwr : m.wrN buf
      ((f.drop count).take (min want ((f.drop count).length))) = some m') :
    X86Env.syscall k m 0 3 buf want 0 0 0 =
      some (.ret (min want ((f.drop count).length))
        { k with fds :=
          [some ⟨f, count + min want ((f.drop count).length)⟩] } m') := by
  simp only [X86Env.syscall]
  rw [if_neg (by decide : ¬ ((3 : Nat) = 0)), if_pos (by decide : 3 ≤ 3)]
  rw [hfd]
  simp only [show (3 : Nat) - 3 = 0 from rfl, List.getElem?_cons_zero]
  rw [kern_count_at_head hhead]
  rw [hwr]
  rfl

/-- One read from the loop head to just after the `jz` (10 steps):
the byte count is `n = min (cap - count) (f.length - count)`, the
oracle is not consumed (its head is a placement, and `Kern.count`
keeps the kernel), the buffer prefix now holds `f.take (count+n)`,
and control is at `loaded`'s jump when `n = 0`, else at the
count-update. -/
theorem loader_read_once {f : List Nat} {A sp0 cap count capB : Nat}
    {c1 c2 buf : Region} {s : X86Int.St}
    (hrip : s.rip = base + 0xd7) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr0 : ∃ v, s.reg 0 = some v) (hr10 : ∃ v, s.reg 10 = some v)
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hmem : s.mem = [c1, c2, buf]) (hbufbase : buf.base = A)
    (hbufrw : buf.rw = true) (hbufhi : buf.hi = A + cap)
    (hdisj1 : c1.hi ≤ A ∨ A + capB ≤ c1.base)
    (hdisj2 : c2.hi ≤ A ∨ A + capB ≤ c2.base) (hcapB : cap ≤ capB)
    (hcontent : s.mem.rdN A count = some (f.take count))
    (hfd : s.k.fds = [some ⟨f, count⟩])
    (hhead : ∃ B, s.k.oracle.head? = some (.at B))
    (hfB : ∀ b ∈ f, b < 256)
    (hcount : count ≤ f.length) (hcountcap : count < cap)
    (hAcap : A + cap < W) (hcap63 : cap < 2 ^ 63) :
    ∃ s', run stage1Im 10 (Out.ok s) = Out.ok s' ∧
      (s'.rip = base + 0x122 ∧ f.length = count ∨
        s'.rip = base + 0xf1 ∧ count < f.length) ∧
      s'.reg 0 = some (min (cap - count) (f.length - count)) ∧
      s'.reg 3 = some A ∧ s'.reg 12 = some cap ∧
      s'.reg 13 = some count ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      (∃ v, s'.reg 10 = some v) ∧ (∃ v, s'.reg 8 = some v) ∧
      (∃ v, s'.reg 9 = some v) ∧
      s'.k.oracle = s.k.oracle ∧
      s'.k.fds =
        [some ⟨f, count + min (cap - count) (f.length - count)⟩] ∧
      (∃ buf', s'.mem = [c1, c2, buf'] ∧ buf'.base = A ∧
        buf'.hi = A + cap ∧ buf'.rw = true) ∧
      s'.mem.rdN A (count + min (cap - count) (f.length - count)) =
        some (f.take (count + min (cap - count) (f.length - count))) ∧
      s'.regs.length = 16 := by
  have hc := stage1_loaderCode
  obtain ⟨n, hn⟩ : ∃ m : Nat, m = min (cap - count) (f.length - count) :=
    ⟨_, rfl⟩
  rw [← hn]
  have hdrop : (f.drop count).length = f.length - count := by
    rw [List.length_drop]
  have hnmin : (f.drop count).take (min (cap - count)
      ((f.drop count).length)) = (f.drop count).take n := by
    rw [hdrop, ← hn]
  obtain ⟨B, hheadB⟩ := hhead
  obtain ⟨v0, hr0v⟩ := hr0
  obtain ⟨v10, hr10v⟩ := hr10
  obtain ⟨v8, hr8v⟩ := hr8
  obtain ⟨v9, hr9v⟩ := hr9
  have hbslen : ((f.drop count).take n).length = n := by
    rw [List.length_take, hn, hdrop]
    omega
  have hdwr1 : c1.hi ≤ buf.base ∨ buf.hi ≤ c1.base := by
    rcases hdisj1 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  have hdwr2 : c2.hi ≤ buf.base ∨ buf.hi ≤ c2.base := by
    rcases hdisj2 with h | h
    · exact Or.inl (by omega)
    · exact Or.inr (by omega)
  obtain ⟨buf', hwr, hbuf'b, hbuf'hi, hbuf'rw⟩ :=
    wrN_third_some (c1 := c1) (c2 := c2) (r := buf) (a := A + count)
      ((f.drop count).take n) hdwr1 hdwr2 hbufrw (by omega) (by
      rw [hbslen, hbufhi]
      have hnn : n ≤ cap - count := by rw [hn]; exact Nat.min_le_left _ _
      omega)
  have hwr' : s.mem.wrN (A + count) ((f.drop count).take n) =
      some [c1, c2, buf'] := by
    rwa [hmem] 
  have hsc : X86Env.syscall s.k s.mem 0 3 (A + count) (cap - count) 0 0 0 =
      some (.ret n
        { s.k with fds := [some ⟨f, count + n⟩] } [c1, c2, buf']) := by
    have h := syscall_read_file (k := s.k) (m := s.mem)
      (m' := [c1, c2, buf']) (f := f) (buf := A + count)
      (count := count) (want := cap - count) (A := B) hfd hheadB (by
      rwa [hnmin])
    rwa [hdrop, ← hn] at h
  have hrdN' : Mem.rdN [c1, c2, buf'] A (count + n) =
      some (f.take (count + n)) := by
    have hpre : Mem.rdN [c1, c2, buf'] A count = some (f.take count) := by
      have h := rdN_wrN_prefix ((f.drop count).take n) hwr'
      rwa [hcontent] at h
    have hnew : Mem.rdN [c1, c2, buf'] (A + count) n =
        some ((f.drop count).take n) := by
      have h := rdN_wrN_same ((f.drop count).take n) s.mem (A + count) hwr'
      rw [hbslen] at h
      have hmapid : ∀ l : List Nat, (∀ b ∈ l, b < 256) →
          l.map (· % 256) = l := by
        intro l hl
        induction l with
        | nil => rfl
        | cons a l ih =>
          simp only [List.map_cons]
          rw [Nat.mod_eq_of_lt (hl a (by simp)),
            ih (fun b hb => hl b (List.mem_cons_of_mem a hb))]
      have hmap : ((f.drop count).take n).map (· % 256) =
          (f.drop count).take n :=
        hmapid _ (fun b hb => hfB b
          (List.mem_of_mem_drop (List.mem_of_mem_take hb)))
      rwa [hmap] at h
    have hcomb := rdN_combine (m := [c1, c2, buf']) (a := A) (n := count)
      (xs := f.take count) (by rw [List.length_take]; omega) hpre
      (k := n) (ys := (f.drop count).take n) hbslen hnew
    rw [List.take_add]
    exact hcomb
  -- steps 1-2: mov rsi, rbx; add rsi, r13
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 6 (some A) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.dd7) hr3
  obtain ⟨d1, hd1⟩ : ∃ t : X86Int.St, t =
      { s.set 6 (some A) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have hd1rip : d1.rip = base + 0xda := by
    rw [hd1]; show s.rip + 3 = base + 0xda; rw [hrip]
  have hlen1 : d1.regs.length = 16 := by
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen16
  have hd1r6 : d1.reg 6 = some A := by
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 6 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hd1r13 : d1.reg 13 = some count := by
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 6)]
    exact hr13
  have e2 : X86Int.step stage1Im d1 =
      .ok { d1.set 6 (some (aluF .add A count).1) with
        fl := (aluF .add A count).2, rip := d1.rip + 3 } :=
    step_alu_add (by rw [hd1rip]; exact hc.dda) hd1r6 hd1r13
  obtain ⟨d2, hd2⟩ : ∃ t : X86Int.St, t =
      { d1.set 6 (some (aluF .add A count).1) with
        fl := (aluF .add A count).2, rip := d1.rip + 3 } := ⟨_, rfl⟩
  have hd2rip : d2.rip = base + 0xdd := by
    rw [hd2]; show d1.rip + 3 = base + 0xdd; rw [hd1rip]
  have hlen2 : d2.regs.length = 16 := by
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).regs.length = 16
    rw [len_set]
    exact hlen1
  have hd2r6 : d2.reg 6 = some (A + count) := by
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 6 = some (A + count)
    rw [reg_set_self (by omega), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega)]
  -- steps 3-4: mov rdx, r12; sub rdx, r13
  have hd2r12 : d2.reg 12 = some cap := by
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 6)]
    exact hr12
  have e3 : X86Int.step stage1Im d2 =
      .ok { d2.set 2 (some cap) with rip := d2.rip + 3 } :=
    step_mov (by rw [hd2rip]; exact hc.ddd) hd2r12
  obtain ⟨d3, hd3⟩ : ∃ t : X86Int.St, t =
      { d2.set 2 (some cap) with rip := d2.rip + 3 } := ⟨_, rfl⟩
  have hd3rip : d3.rip = base + 0xe0 := by
    rw [hd3]; show d2.rip + 3 = base + 0xe0; rw [hd2rip]
  have hlen3 : d3.regs.length = 16 := by
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen2
  have hd3r2 : d3.reg 2 = some cap := by
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hd3r13 : d3.reg 13 = some count := by
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 6)]
    exact hd1r13
  have e4 : X86Int.step stage1Im d3 =
      .ok { d3.set 2 (some (aluF .sub cap count).1) with
        fl := (aluF .sub cap count).2, rip := d3.rip + 3 } :=
    step_alu_sub (by rw [hd3rip]; exact hc.de0) hd3r2 hd3r13
  obtain ⟨d4, hd4⟩ : ∃ t : X86Int.St, t =
      { d3.set 2 (some (aluF .sub cap count).1) with
        fl := (aluF .sub cap count).2, rip := d3.rip + 3 } := ⟨_, rfl⟩
  have hd4rip : d4.rip = base + 0xe3 := by
    rw [hd4]; show d3.rip + 3 = base + 0xe3; rw [hd3rip]
  have hlen4 : d4.regs.length = 16 := by
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).regs.length = 16
    rw [len_set]
    exact hlen3
  have hd4r2 : d4.reg 2 = some (cap - count) := by
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 2 = some (cap - count)
    rw [reg_set_self (by omega), aluF_sub_fst]
    have h1 : cap + W - count = (cap - count) + W := by omega
    rw [h1, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  -- steps 5-6: xor eax, eax; mov edi, r14d
  have hd4r0 : d4.reg 0 = some v0 := by
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 0 = some v0
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 0 = some v0
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 0 = some v0
    rw [reg_set_other (by decide : (0 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 0 = some v0
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact hr0v
  have e5 : X86Int.step stage1Im d4 =
      .ok { d4.set 0 (some (aluF .xor v0 v0).1) with
        fl := (aluF .xor v0 v0).2, rip := d4.rip + 2 } :=
    step_alu_xor (by rw [hd4rip]; exact hc.de3) hd4r0 hd4r0
  obtain ⟨d5, hd5⟩ : ∃ t : X86Int.St, t =
      { d4.set 0 (some (aluF .xor v0 v0).1) with
        fl := (aluF .xor v0 v0).2, rip := d4.rip + 2 } := ⟨_, rfl⟩
  have hd5rip : d5.rip = base + 0xe5 := by
    rw [hd5]; show d4.rip + 2 = base + 0xe5; rw [hd4rip]
  have hlen5 : d5.regs.length = 16 := by
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).regs.length = 16
    rw [len_set]
    exact hlen4
  have hd5r0 : d5.reg 0 = some 0 := by
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 0 = some 0
    rw [reg_set_self (by omega), aluF_xor_fst_self]
  have hd5r14 : d5.reg 14 = some 3 := by
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    exact hr14
  have e6 : X86Int.step stage1Im d5 =
      .ok { d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 } :=
    step_mov32 (by rw [hd5rip]; exact hc.de5) hd5r14
  obtain ⟨d6, hd6⟩ : ∃ t : X86Int.St, t =
      { d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 } := ⟨_, rfl⟩
  have hd6rip : d6.rip = base + 0xe8 := by
    rw [hd6]; show d5.rip + 3 = base + 0xe8; rw [hd5rip]
  have hlen6 : d6.regs.length = 16 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with
      rip := d5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]
    exact hlen5
  have hd6r7 : d6.reg 7 = some 3 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 7 =
      some 3
    rw [reg_with_rip, reg_set_self (by omega)]
  have hd6r0 : d6.reg 0 = some 0 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 0 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact hd5r0
  have hd6r6 : d6.reg 6 = some (A + count) := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 6 =
      some (A + count)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 6 = some (A + count)
    rw [reg_set_other (by decide : (6 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 6 = some (A + count)
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 6 =
      some (A + count)
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact hd2r6
  have hd6r2 : d6.reg 2 = some (cap - count) := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 2 =
      some (cap - count)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 2 = some (cap - count)
    rw [reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact hd4r2
  have hd6r10 : d6.reg 10 = some v10 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 10 =
      some v10
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 10 = some v10
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 10 = some v10
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 6)]
    exact hr10v
  have hd6r8 : d6.reg 8 = some v8 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6)]
    exact hr8v
  have hd6r9 : d6.reg 9 = some v9 := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    exact hr9v
  have hd6mem : d6.mem = s.mem := by
    rw [hd6, hd5, hd4, hd3, hd2, hd1]
    rfl
  have hd6k : d6.k = s.k := by
    rw [hd6, hd5, hd4, hd3, hd2, hd1]
    rfl
  have hd6r3 : d6.reg 3 = some A := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 6)]
    exact hr3
  have hd6r12 : d6.reg 12 = some cap := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 2)]
    exact hd2r12
  have hd6r13 : d6.reg 13 = some count := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]
    exact hd3r13
  have hd6r4 : d6.reg 4 = some (sp0 + 24) := by
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    rw [hd5]
    show (d4.set 0 (some (aluF .xor v0 v0).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 0)]
    rw [hd4]
    show (d3.set 2 (some (aluF .sub cap count).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hd3]
    show ({ d2.set 2 (some cap) with rip := d2.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hd2]
    show (d1.set 6 (some (aluF .add A count).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 6)]
    rw [hd1]
    show ({ s.set 6 (some A) with rip := s.rip + 3 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    exact hr4
  -- step 7: read(3, buf + count, cap - count)
  have hsc6 : X86Env.syscall d6.k d6.mem 0 3 (A + count) (cap - count)
        0 0 0 =
      some (.ret n { s.k with fds := [some ⟨f, count + n⟩] }
        [c1, c2, buf']) := by
    rw [hd6k, hd6mem]
    exact hsc
  have e7 : X86Int.step stage1Im d6 =
      .ok { ((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11 none with
        k := { s.k with fds := [some ⟨f, count + n⟩] },
        mem := [c1, c2, buf'], rip := d6.rip + 2 } :=
    step_syscall_ret (by rw [hd6rip]; exact hc.de8) hd6r0 hd6r7 hd6r6
      hd6r2 hd6r10 hd6r8 hd6r9 hsc6
  obtain ⟨d7, hd7⟩ : ∃ t : X86Int.St, t =
      { ((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11 none with
        k := { s.k with fds := [some ⟨f, count + n⟩] },
        mem := [c1, c2, buf'], rip := d6.rip + 2 } := ⟨_, rfl⟩
  have hd7rip : d7.rip = base + 0xea := by
    rw [hd7]; show d6.rip + 2 = base + 0xea; rw [hd6rip]
  have hlen7 : d7.regs.length = 16 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]
    exact hlen6
  have hd7r0 : d7.reg 0 = some n := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 0 = some n
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hd7r3 : d7.reg 3 = some A := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 11),
      reg_set_other (by decide : (3 : Nat) ≠ 1),
      reg_set_other (by decide : (3 : Nat) ≠ 0)]
    exact hd6r3
  have hd7r12 : d7.reg 12 = some cap := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 11),
      reg_set_other (by decide : (12 : Nat) ≠ 1),
      reg_set_other (by decide : (12 : Nat) ≠ 0)]
    exact hd6r12
  have hd7r13 : d7.reg 13 = some count := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 11),
      reg_set_other (by decide : (13 : Nat) ≠ 1),
      reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hd6r13
  have hd7r14 : d7.reg 14 = some 3 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 11),
      reg_set_other (by decide : (14 : Nat) ≠ 1),
      reg_set_other (by decide : (14 : Nat) ≠ 0)]
    rw [hd6]
    show ({ d5.set 7 (some (3 % 2 ^ 32)) with rip := d5.rip + 3 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 7)]
    exact hd5r14
  have hd7r4 : d7.reg 4 = some (sp0 + 24) := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hd6r4
  have hd7r10 : d7.reg 10 = some v10 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 10 = some v10
    rw [reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact hd6r10
  have hd7r8 : d7.reg 8 = some v8 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hd6r8
  have hd7r9 : d7.reg 9 = some v9 := by
    rw [hd7]
    show (((d6.set 0 (some n)).set 1 (some (d6.rip + 2))).set 11
      none).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hd6r9
  have hd7mem : d7.mem = [c1, c2, buf'] := by rw [hd7]
  have hd7k : d7.k = { s.k with fds := [some ⟨f, count + n⟩] } := by
    rw [hd7]
  -- step 8: test rax, rax
  have e8 : X86Int.step stage1Im d7 =
      .ok { d7 with fl := (aluF .and n n).2, rip := d7.rip + 3 } :=
    step_test (by rw [hd7rip]; exact hc.dea) hd7r0 hd7r0
  obtain ⟨d8, hd8⟩ : ∃ t : X86Int.St, t =
      { d7 with fl := (aluF .and n n).2, rip := d7.rip + 3 } := ⟨_, rfl⟩
  have hd8rip : d8.rip = base + 0xed := by
    rw [hd8]; show d7.rip + 3 = base + 0xed; rw [hd7rip]
  have hn63 : n < 2 ^ 63 := by
    have hle : n ≤ cap - count := by
      rw [hn]; exact Nat.min_le_left _ _
    omega
  have hd8sf : d8.fl.sf = some false := by
    rw [hd8]
    show (aluF .and n n).2.sf = some false
    have hland : Nat.land n n = n := by simp
    have hmsb : X86Int.msb n = false := by
      simp only [X86Int.msb]
      rw [Nat.div_eq_of_lt hn63]
      decide
    simp [aluF, hland, hmsb]
  have hd8zf : d8.fl.zf = some (n == 0) := by
    rw [hd8]
    show (aluF .and n n).2.zf = some (n == 0)
    rw [aluF_zf, aluF_and_fst_self]
  have hd8r0 : d8.reg 0 = some n := by rw [hd8]; exact hd7r0
  have hd8r3 : d8.reg 3 = some A := by rw [hd8]; exact hd7r3
  have hd8r12 : d8.reg 12 = some cap := by rw [hd8]; exact hd7r12
  have hd8r13 : d8.reg 13 = some count := by rw [hd8]; exact hd7r13
  have hd8r14 : d8.reg 14 = some 3 := by rw [hd8]; exact hd7r14
  have hd8r4 : d8.reg 4 = some (sp0 + 24) := by rw [hd8]; exact hd7r4
  have hd8r10 : d8.reg 10 = some v10 := by rw [hd8]; exact hd7r10
  have hd8r8 : d8.reg 8 = some v8 := by rw [hd8]; exact hd7r8
  have hd8r9 : d8.reg 9 = some v9 := by rw [hd8]; exact hd7r9
  have hd8mem : d8.mem = [c1, c2, buf'] := by rw [hd8]; exact hd7mem
  have hd8k : d8.k = { s.k with fds := [some ⟨f, count + n⟩] } := by rw [hd8]; exact hd7k
  have hlen8 : d8.regs.length = 16 := by rw [hd8]; exact hlen7
  -- step 9: js fail1 (not taken: n ≥ 0)
  have hd8fl8 : X86Int.cond d8.fl 8 = some false :=
    cond_cc8_of_sf_false hd8sf
  have e9 : X86Int.step stage1Im d8 =
      .ok { d8 with rip := d8.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := d8)
      (by rw [hd8rip]; exact hc.ded) hd8fl8
    simpa using h
  obtain ⟨d9, hd9⟩ : ∃ t : X86Int.St, t =
      { d8 with rip := d8.rip + 2 } := ⟨_, rfl⟩
  have hd9rip : d9.rip = base + 0xef := by
    rw [hd9]; show d8.rip + 2 = base + 0xef; rw [hd8rip]
  have hd9r0 : d9.reg 0 = some n := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 0 = some n
    rw [reg_with_rip]; exact hd8r0
  have hd9r3 : d9.reg 3 = some A := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 3 = some A
    rw [reg_with_rip]; exact hd8r3
  have hd9r12 : d9.reg 12 = some cap := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 12 = some cap
    rw [reg_with_rip]; exact hd8r12
  have hd9r13 : d9.reg 13 = some count := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 13 = some count
    rw [reg_with_rip]; exact hd8r13
  have hd9r14 : d9.reg 14 = some 3 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 14 = some 3
    rw [reg_with_rip]; exact hd8r14
  have hd9r4 : d9.reg 4 = some (sp0 + 24) := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hd8r4
  have hd9r10 : d9.reg 10 = some v10 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 10 = some v10
    rw [reg_with_rip]; exact hd8r10
  have hd9r8 : d9.reg 8 = some v8 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 8 = some v8
    rw [reg_with_rip]; exact hd8r8
  have hd9r9 : d9.reg 9 = some v9 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).reg 9 = some v9
    rw [reg_with_rip]; exact hd8r9
  have hd9mem : d9.mem = [c1, c2, buf'] := by rw [hd9]; exact hd8mem
  have hd9k : d9.k = { s.k with fds := [some ⟨f, count + n⟩] } := by rw [hd9]; exact hd8k
  have hlen9 : d9.regs.length = 16 := by
    rw [hd9]; show ({ d8 with rip := d8.rip + 2 }).regs.length = 16
    rw [len_with_rip]; exact hlen8
  have hd9fl : d9.fl = d8.fl := by rw [hd9]
  -- step 10: jz loaded / fallthrough on n
  have hd9fl4 : X86Int.cond d9.fl 4 = some (n == 0) := by
    rw [hd9fl]
    exact cond_cc4 hd8zf
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok d1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hd1, run_zero]
  have r2 : run stage1Im 1 (Out.ok d1) = Out.ok d2 := by
    rw [run_succ_ok stage1Im 0 d1, e2, ← hd2, run_zero]
  have r3 : run stage1Im 1 (Out.ok d2) = Out.ok d3 := by
    rw [run_succ_ok stage1Im 0 d2, e3, ← hd3, run_zero]
  have r4 : run stage1Im 1 (Out.ok d3) = Out.ok d4 := by
    rw [run_succ_ok stage1Im 0 d3, e4, ← hd4, run_zero]
  have r5 : run stage1Im 1 (Out.ok d4) = Out.ok d5 := by
    rw [run_succ_ok stage1Im 0 d4, e5, ← hd5, run_zero]
  have r6 : run stage1Im 1 (Out.ok d5) = Out.ok d6 := by
    rw [run_succ_ok stage1Im 0 d5, e6, ← hd6, run_zero]
  have r7 : run stage1Im 1 (Out.ok d6) = Out.ok d7 := by
    rw [run_succ_ok stage1Im 0 d6, e7, ← hd7, run_zero]
  have r8 : run stage1Im 1 (Out.ok d7) = Out.ok d8 := by
    rw [run_succ_ok stage1Im 0 d7, e8, ← hd8, run_zero]
  have r9 : run stage1Im 1 (Out.ok d8) = Out.ok d9 := by
    rw [run_succ_ok stage1Im 0 d8, e9, ← hd9, run_zero]
  by_cases hn0 : n = 0
  · have hfl : X86Int.cond d9.fl 4 = some true := by
      rw [hd9fl4, hn0]
      decide
    have e10 : X86Int.step stage1Im d9 =
        .ok { d9 with rip := base + 0x122 } := by
      have h := step_jcc (im := stage1Im) (s := d9)
        (by rw [hd9rip]; exact hc.def_) hfl
      simpa using h
    obtain ⟨d10, hd10⟩ : ∃ t : X86Int.St, t =
        { d9 with rip := base + 0x122 } := ⟨_, rfl⟩
    have r10 : run stage1Im 1 (Out.ok d9) = Out.ok d10 := by
      rw [run_succ_ok stage1Im 0 d9, e10, ← hd10, run_zero]
    have hrun : run stage1Im 10 (Out.ok s) = Out.ok d10 := by
      rw [show (10 : Nat) = 1 + 9 from rfl, run_add stage1Im 1 9, r1,
        show (9 : Nat) = 1 + 8 from rfl, run_add stage1Im 1 8, r2,
        show (8 : Nat) = 1 + 7 from rfl, run_add stage1Im 1 7, r3,
        show (7 : Nat) = 1 + 6 from rfl, run_add stage1Im 1 6, r4,
        show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r5,
        show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r6,
        show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r7,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r8,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r9, r10]
    have hfeq : f.length = count := by
      rw [hn] at hn0
      omega
    refine ⟨d10, hrun, Or.inl ⟨by rw [hd10], hfeq⟩, ?_, ?_, ?_, ?_, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ⟨buf', ?_, hbuf'b.trans hbufbase, hbuf'hi.trans hbufhi, hbuf'rw.trans hbufrw⟩, ?_, ?_⟩
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 0 = some n
      rw [reg_with_rip]; exact hd9r0
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 3 = some A
      rw [reg_with_rip]; exact hd9r3
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 12 = some cap
      rw [reg_with_rip]; exact hd9r12
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 13 = some count
      rw [reg_with_rip]; exact hd9r13
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 14 = some 3
      rw [reg_with_rip]; exact hd9r14
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hd9r4
    · refine ⟨v10, ?_⟩
      rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 10 = some v10
      rw [reg_with_rip]; exact hd9r10
    · refine ⟨v8, ?_⟩
      rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 8 = some v8
      rw [reg_with_rip]; exact hd9r8
    · refine ⟨v9, ?_⟩
      rw [hd10]
      show ({ d9 with rip := base + 0x122 }).reg 9 = some v9
      rw [reg_with_rip]; exact hd9r9
    · rw [hd10]
      show d9.k.oracle = s.k.oracle
      rw [hd9k]
    · rw [hd10]
      exact congrArg X86Env.Kern.fds hd9k
    · rw [hd10]
      exact hd9mem
    · rw [hd10]
      show Mem.rdN d9.mem A (count + n) = some (f.take (count + n))
      rw [hd9mem]
      exact hrdN'
    · rw [hd10]
      show ({ d9 with rip := base + 0x122 }).regs.length = 16
      rw [len_with_rip]; exact hlen9
  · have hfl : X86Int.cond d9.fl 4 = some false := by
      rw [hd9fl4]
      simp [hn0]
    have e10 : X86Int.step stage1Im d9 =
        .ok { d9 with rip := d9.rip + 2 } := by
      have h := step_jcc (im := stage1Im) (s := d9)
        (by rw [hd9rip]; exact hc.def_) hfl
      simpa using h
    obtain ⟨d10, hd10⟩ : ∃ t : X86Int.St, t =
        { d9 with rip := d9.rip + 2 } := ⟨_, rfl⟩
    have r10 : run stage1Im 1 (Out.ok d9) = Out.ok d10 := by
      rw [run_succ_ok stage1Im 0 d9, e10, ← hd10, run_zero]
    have hrun : run stage1Im 10 (Out.ok s) = Out.ok d10 := by
      rw [show (10 : Nat) = 1 + 9 from rfl, run_add stage1Im 1 9, r1,
        show (9 : Nat) = 1 + 8 from rfl, run_add stage1Im 1 8, r2,
        show (8 : Nat) = 1 + 7 from rfl, run_add stage1Im 1 7, r3,
        show (7 : Nat) = 1 + 6 from rfl, run_add stage1Im 1 6, r4,
        show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r5,
        show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r6,
        show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r7,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r8,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r9, r10]
    have hflt : count < f.length := by
      rw [hn] at hn0
      omega
    refine ⟨d10, hrun, Or.inr ⟨by
        rw [hd10]; show d9.rip + 2 = base + 0xf1; rw [hd9rip], hflt⟩, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ⟨buf', ?_, hbuf'b.trans hbufbase, hbuf'hi.trans hbufhi, hbuf'rw.trans hbufrw⟩,
      ?_, ?_⟩
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 0 = some n
      rw [reg_with_rip]; exact hd9r0
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 3 = some A
      rw [reg_with_rip]; exact hd9r3
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 12 = some cap
      rw [reg_with_rip]; exact hd9r12
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 13 = some count
      rw [reg_with_rip]; exact hd9r13
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 14 = some 3
      rw [reg_with_rip]; exact hd9r14
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hd9r4
    · refine ⟨v10, ?_⟩
      rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 10 = some v10
      rw [reg_with_rip]; exact hd9r10
    · refine ⟨v8, ?_⟩
      rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 8 = some v8
      rw [reg_with_rip]; exact hd9r8
    · refine ⟨v9, ?_⟩
      rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).reg 9 = some v9
      rw [reg_with_rip]; exact hd9r9
    · rw [hd10]
      show d9.k.oracle = s.k.oracle
      rw [hd9k]
    · rw [hd10]
      exact congrArg X86Env.Kern.fds hd9k
    · rw [hd10]
      exact hd9mem
    · rw [hd10]
      show Mem.rdN d9.mem A (count + n) = some (f.take (count + n))
      rw [hd9mem]
      exact hrdN'
    · rw [hd10]
      show ({ d9 with rip := d9.rip + 2 }).regs.length = 16
      rw [len_with_rip]; exact hlen9

/-- The count update after a positive read (4 steps from
`base + 0xf1`): `count += n`, then back to the loop head while
room remains, else fall through to the growth block at
`base + 0xfc` with the buffer exactly full. -/
theorem loader_count_update {A sp0 cap count n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0xf1) (hlen16 : s.regs.length = 16)
    (hr0 : s.reg 0 = some n) (hr3 : s.reg 3 = some A)
    (hr12 : s.reg 12 = some cap) (hr13 : s.reg 13 = some count)
    (hr14 : s.reg 14 = some 3) (hr4 : s.reg 4 = some (sp0 + 24))
    (hr10 : ∃ v, s.reg 10 = some v) (hr8 : ∃ v, s.reg 8 = some v)
    (hr9 : ∃ v, s.reg 9 = some v)
    (hnle : count + n ≤ cap) (hcapW : cap < W) :
    ∃ s', run stage1Im 4 (Out.ok s) = Out.ok s' ∧
      (s'.rip = base + 0xd7 ∧ count + n < cap ∨
        s'.rip = base + 0xfc ∧ count + n = cap) ∧
      s'.reg 13 = some (count + n) ∧ s'.reg 3 = some A ∧
      s'.reg 12 = some cap ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
      (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      s'.mem = s.mem ∧ s'.k = s.k ∧ s'.regs.length = 16 := by
  have hc := stage1_loaderCode
  have hadd : (aluF .add count n).1 = count + n := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  -- step 1: add r13, rax
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 13 (some (aluF .add count n).1) with
        fl := (aluF .add count n).2, rip := s.rip + 3 } :=
    step_alu_add (by rw [hrip]; exact hc.df1) hr13 hr0
  obtain ⟨u1, hu1⟩ : ∃ t : X86Int.St, t =
      { s.set 13 (some (aluF .add count n).1) with
        fl := (aluF .add count n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have hu1rip : u1.rip = base + 0xf4 := by
    rw [hu1]; show s.rip + 3 = base + 0xf4; rw [hrip]
  have hlen1 : u1.regs.length = 16 := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).regs.length = 16
    rw [len_set]; exact hlen16
  have hu1r13 : u1.reg 13 = some (count + n) := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 13 = some (count + n)
    rw [reg_set_self (by omega), hadd]
  have hu1r12 : u1.reg 12 = some cap := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 13)]; exact hr12
  have hu1r3 : u1.reg 3 = some A := by
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 13)]; exact hr3
  have hu1mem : u1.mem = s.mem := by rw [hu1]; rfl
  have hu1k : u1.k = s.k := by rw [hu1]; rfl
  -- step 2: mov rdx, r12
  have e2 : X86Int.step stage1Im u1 =
      .ok { u1.set 2 (some cap) with rip := u1.rip + 3 } :=
    step_mov (by rw [hu1rip]; exact hc.df4) hu1r12
  obtain ⟨u2, hu2⟩ : ∃ t : X86Int.St, t =
      { u1.set 2 (some cap) with rip := u1.rip + 3 } := ⟨_, rfl⟩
  have hu2rip : u2.rip = base + 0xf7 := by
    rw [hu2]; show u1.rip + 3 = base + 0xf7; rw [hu1rip]
  have hlen2 : u2.regs.length = 16 := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen1
  have hu2r2 : u2.reg 2 = some cap := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hu2r13 : u2.reg 13 = some (count + n) := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 13 =
      some (count + n)
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    exact hu1r13
  have hu2r3 : u2.reg 3 = some A := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 2)]
    exact hu1r3
  have hu2r12 : u2.reg 12 = some cap := by
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 2)]
    exact hu1r12
  have hu2mem : u2.mem = s.mem := by rw [hu2]; exact hu1mem
  have hu2k : u2.k = s.k := by rw [hu2]; exact hu1k
  -- step 3: sub rdx, r13
  have hsub : (aluF .sub cap (count + n)).1 = cap - (count + n) := by
    rw [aluF_sub_fst,
      show cap + W - (count + n) = (cap - (count + n)) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have e3 : X86Int.step stage1Im u2 =
      .ok { u2.set 2 (some (aluF .sub cap (count + n)).1) with
        fl := (aluF .sub cap (count + n)).2, rip := u2.rip + 3 } :=
    step_alu_sub (by rw [hu2rip]; exact hc.df7) hu2r2 hu2r13
  obtain ⟨u3, hu3⟩ : ∃ t : X86Int.St, t =
      { u2.set 2 (some (aluF .sub cap (count + n)).1) with
        fl := (aluF .sub cap (count + n)).2, rip := u2.rip + 3 } :=
    ⟨_, rfl⟩
  have hu3rip : u3.rip = base + 0xfa := by
    rw [hu3]; show u2.rip + 3 = base + 0xfa; rw [hu2rip]
  have hlen3 : u3.regs.length = 16 := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).regs.length = 16
    rw [len_set]; exact hlen2
  have hu3zf : u3.fl.zf = some (cap - (count + n) == 0) := by
    rw [hu3]
    show (aluF .sub cap (count + n)).2.zf = some (cap - (count + n) == 0)
    rw [aluF_zf, hsub]
  have hu3r13 : u3.reg 13 = some (count + n) := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 13 =
      some (count + n)
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]; exact hu2r13
  have hu3r3 : u3.reg 3 = some A := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 3 = some A
    rw [reg_set_other (by decide : (3 : Nat) ≠ 2)]; exact hu2r3
  have hu3r12 : u3.reg 12 = some cap := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 12 = some cap
    rw [reg_set_other (by decide : (12 : Nat) ≠ 2)]; exact hu2r12
  have hu3r14 : u3.reg 14 = some 3 := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 13)]; exact hr14
  have hu3r4 : u3.reg 4 = some (sp0 + 24) := by
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 4 =
      some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 13)]; exact hr4
  have hu3r0 : ∃ v, u3.reg 0 = some v := by
    refine ⟨n, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 0 = some n
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 0 = some n
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 0 = some n
    rw [reg_set_other (by decide : (0 : Nat) ≠ 13)]; exact hr0
  have hu3r10 : ∃ v, u3.reg 10 = some v := by
    obtain ⟨v, hv⟩ := hr10
    refine ⟨v, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 10 = some v
    rw [reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 10 = some v
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 10 = some v
    rw [reg_set_other (by decide : (10 : Nat) ≠ 13)]; exact hv
  have hu3r8 : ∃ v, u3.reg 8 = some v := by
    obtain ⟨v, hv⟩ := hr8
    refine ⟨v, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 8 = some v
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 8 = some v
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 8 = some v
    rw [reg_set_other (by decide : (8 : Nat) ≠ 13)]; exact hv
  have hu3r9 : ∃ v, u3.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    refine ⟨v, ?_⟩
    rw [hu3]
    show (u2.set 2 (some (aluF .sub cap (count + n)).1)).reg 9 = some v
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hu2]
    show ({ u1.set 2 (some cap) with rip := u1.rip + 3 }).reg 9 = some v
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hu1]
    show (s.set 13 (some (aluF .add count n).1)).reg 9 = some v
    rw [reg_set_other (by decide : (9 : Nat) ≠ 13)]; exact hv
  have hu3mem : u3.mem = s.mem := by rw [hu3]; exact hu2mem
  have hu3k : u3.k = s.k := by rw [hu3]; exact hu2k
  -- step 4: jnz readloop / fall through to growth
  have hu3fl5 : X86Int.cond u3.fl 5 = some (!(cap - (count + n) == 0)) :=
    cond_cc5 hu3zf
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok u1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hu1, run_zero]
  have r2 : run stage1Im 1 (Out.ok u1) = Out.ok u2 := by
    rw [run_succ_ok stage1Im 0 u1, e2, ← hu2, run_zero]
  have r3 : run stage1Im 1 (Out.ok u2) = Out.ok u3 := by
    rw [run_succ_ok stage1Im 0 u2, e3, ← hu3, run_zero]
  by_cases hfull : count + n = cap
  · have hfl : X86Int.cond u3.fl 5 = some false := by
      rw [hu3fl5]
      simp [show cap - (count + n) = 0 by omega]
    have e4 : X86Int.step stage1Im u3 =
        .ok { u3 with rip := u3.rip + 2 } := by
      have h := step_jcc (im := stage1Im) (s := u3)
        (by rw [hu3rip]; exact hc.dfa) hfl
      simpa using h
    obtain ⟨u4, hu4⟩ : ∃ t : X86Int.St, t =
        { u3 with rip := u3.rip + 2 } := ⟨_, rfl⟩
    have r4 : run stage1Im 1 (Out.ok u3) = Out.ok u4 := by
      rw [run_succ_ok stage1Im 0 u3, e4, ← hu4, run_zero]
    have hrun : run stage1Im 4 (Out.ok s) = Out.ok u4 := by
      rw [show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r1,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r2,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r3, r4]
    refine ⟨u4, hrun, Or.inr ⟨by
        rw [hu4]; show u3.rip + 2 = base + 0xfc; rw [hu3rip], hfull⟩, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 13 = some (count + n)
      rw [reg_with_rip]; exact hu3r13
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 3 = some A
      rw [reg_with_rip]; exact hu3r3
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 12 = some cap
      rw [reg_with_rip]; exact hu3r12
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 14 = some 3
      rw [reg_with_rip]; exact hu3r14
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hu3r4
    · obtain ⟨v, hv⟩ := hu3r0
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 0 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r10
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 10 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r8
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 8 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r9
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).reg 9 = some v
      rw [reg_with_rip]; exact hv
    · rw [hu4]; exact hu3mem
    · rw [hu4]; exact hu3k
    · rw [hu4]
      show ({ u3 with rip := u3.rip + 2 }).regs.length = 16
      rw [len_with_rip]; exact hlen3
  · have hfl : X86Int.cond u3.fl 5 = some true := by
      rw [hu3fl5]
      simp [show cap - (count + n) ≠ 0 by omega]
    have e4 : X86Int.step stage1Im u3 =
        .ok { u3 with rip := base + 0xd7 } := by
      have h := step_jcc (im := stage1Im) (s := u3)
        (by rw [hu3rip]; exact hc.dfa) hfl
      simpa using h
    obtain ⟨u4, hu4⟩ : ∃ t : X86Int.St, t =
        { u3 with rip := base + 0xd7 } := ⟨_, rfl⟩
    have r4 : run stage1Im 1 (Out.ok u3) = Out.ok u4 := by
      rw [run_succ_ok stage1Im 0 u3, e4, ← hu4, run_zero]
    have hrun : run stage1Im 4 (Out.ok s) = Out.ok u4 := by
      rw [show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r1,
        show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r2,
        show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r3, r4]
    refine ⟨u4, hrun, Or.inl ⟨by rw [hu4], by omega⟩, ?_, ?_, ?_, ?_,
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 13 = some (count + n)
      rw [reg_with_rip]; exact hu3r13
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 3 = some A
      rw [reg_with_rip]; exact hu3r3
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 12 = some cap
      rw [reg_with_rip]; exact hu3r12
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 14 = some 3
      rw [reg_with_rip]; exact hu3r14
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hu3r4
    · obtain ⟨v, hv⟩ := hu3r0
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 0 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r10
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 10 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r8
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 8 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hu3r9
      refine ⟨v, ?_⟩
      rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).reg 9 = some v
      rw [reg_with_rip]; exact hv
    · rw [hu4]; exact hu3mem
    · rw [hu4]; exact hu3k
    · rw [hu4]
      show ({ u3 with rip := base + 0xd7 }).regs.length = 16
      rw [len_with_rip]; exact hlen3

theorem pages_eq_of_mod_zero {n : Nat} (h : n % page = 0) :
    X86Env.pages n = n := by
  have hp : page = 4096 := rfl
  obtain ⟨q, hq⟩ := Nat.dvd_of_mod_eq_zero h
  unfold X86Env.pages
  rw [hq, hp]
  omega

theorem find_third {c1 c2 r : Region} {A : Nat}
    (h1 : c1.hi ≤ A ∨ A < c1.base) (h2 : c2.hi ≤ A ∨ A < c2.base)
    (hge : r.base ≤ A)
    (hlt : A < r.hi) :
    Mem.find [c1, c2, r] A 0 = some (2, r) := by
  have hnot1 : ¬ (c1.base ≤ A ∧ A < c1.hi) := by
    rcases h1 with h | h <;> omega
  have hnot2 : ¬ (c2.base ≤ A ∧ A < c2.hi) := by
    rcases h2 with h | h <;> omega
  simp [Mem.find, hnot1, hnot2, hge, hlt]

theorem free_pair {c1 c2 : Region} {A n : Nat}
    (h1 : c1.hi ≤ A ∨ A + n ≤ c1.base)
    (h2 : c2.hi ≤ A ∨ A + n ≤ c2.base) :
    Mem.free [c1, c2] A n = true := by
  rw [Mem.free, List.all_eq_true]
  intro r hr
  simp at hr
  rcases hr with rfl | rfl <;>
    simp only [Bool.or_eq_true, decide_eq_true_eq]
  · rcases h1 with h | h
    · exact Or.inr h
    · exact Or.inl h
  · rcases h2 with h | h
    · exact Or.inr h
    · exact Or.inl h

theorem mem_rd_third {c1 c2 r : Region} {a v : Nat}
    (h1 : c1.hi ≤ a ∨ a < c1.base) (h2 : c2.hi ≤ a ∨ a < c2.base)
    (hge : r.base ≤ a)
    (hbyte : r.data[a - r.base]? = some v) :
    Mem.rd [c1, c2, r] a = some v := by
  have hlt : a < r.hi := by
    have hh := (List.getElem?_eq_some_iff.1 hbyte).1
    unfold Region.hi
    omega
  have hnot1 : ¬ (c1.base ≤ a ∧ a < c1.hi) := by
    rcases h1 with h | h <;> omega
  have hnot2 : ¬ (c2.base ≤ a ∧ a < c2.hi) := by
    rcases h2 with h | h <;> omega
  simp [Mem.rd, Mem.find, hnot1, hnot2, hge, hlt, hbyte]

theorem rdN_third_take {c1 c2 r : Region} {A k capB : Nat}
    (h1 : c1.hi ≤ A ∨ A + capB ≤ c1.base)
    (h2 : c2.hi ≤ A ∨ A + capB ≤ c2.base) (hb : r.base = A)
    (hk : k ≤ r.data.length) (hkB : k ≤ capB) :
    Mem.rdN [c1, c2, r] A k = some (r.data.take k) := by
  apply rdN_of_rd (by rw [List.length_take, Nat.min_eq_left hk])
  intro i hi
  have hilen : i < r.data.length := by omega
  have hgi : r.data[i]? = some r.data[i] := List.getElem?_eq_getElem hilen
  refine ⟨r.data[i], ?_, ?_⟩
  · have hbyte : r.data[(A + i) - r.base]? = some r.data[i] := by
      rw [hb, show A + i - A = i by omega]; exact hgi
    have hd1 : c1.hi ≤ A + i ∨ A + i < c1.base := by
      rcases h1 with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    have hd2 : c2.hi ≤ A + i ∨ A + i < c2.base := by
      rcases h2 with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    exact mem_rd_third hd1 hd2 (by omega) hbyte
  · rw [List.getElem?_take]
    simp [hi, hgi]

theorem rdN_grow_preserve {c1 c2 r : Region} {A k z capB : Nat}
    {L : List Nat}
    (h1 : c1.hi ≤ A ∨ A + capB ≤ c1.base)
    (h2 : c2.hi ≤ A ∨ A + capB ≤ c2.base) (hb : r.base = A)
    (hk : k ≤ r.data.length) (hkB : k ≤ capB)
    (hold : Mem.rdN [c1, c2, r] A k = some L) :
    Mem.rdN [c1, c2, ⟨A, r.data ++ List.replicate z 0, true⟩] A k =
      some L := by
  have hlen : (r.data ++ List.replicate z 0).length =
      r.data.length + z := by
    rw [List.length_append, List.length_replicate]
  have hk' : k ≤ (r.data ++ List.replicate z 0).length := by
    rw [hlen]; omega
  have hnew := rdN_third_take (r := ⟨A, r.data ++ List.replicate z 0, true⟩)
    h1 h2 rfl hk' hkB
  have htake : (r.data ++ List.replicate z 0).take k = r.data.take k := by
    rw [List.take_append, show k - r.data.length = 0 by omega,
      List.take_zero, List.append_nil]
  rw [htake] at hnew
  have hold' := rdN_third_take h1 h2 hb hk hkB
  rw [hold'] at hold
  injection hold with hL
  rw [← hL]
  exact hnew

theorem syscall_mremap_grow {k : Kern} {c1 c2 r : Region}
    {A cap a5 a6 : Nat} {rest : List Choice}
    (hfind : Mem.find [c1, c2, r] A 0 = some (2, r))
    (hbase : r.base = A) (hlen : r.data.length = cap)
    (hrw : r.rw = true)
    (hpages : X86Env.pages cap = cap)
    (hpages2 : X86Env.pages (cap + cap) = cap + cap)
    (horacle : k.oracle = .at A :: rest)
    (hplace : X86Env.place [c1, c2] (some (.at A)) (cap + cap) =
      some (some A)) :
    X86Env.syscall k [c1, c2, r] 25 A cap (cap + cap) MREMAP_MAYMOVE
        a5 a6 =
      some (.ret A { k with oracle := rest }
        [c1, c2, ⟨A, r.data ++ List.replicate cap 0, true⟩]) := by
  have hval : ¬ (r.base ≠ A ∨ r.data.length ≠ X86Env.pages cap ∨
      ¬ r.rw) := by simp [hbase, hlen, hrw, hpages]
  have hgrow : ¬ X86Env.pages (cap + cap) < r.data.length := by
    rw [hpages2, hlen]; omega
  have herase : List.eraseIdx [c1, c2, r] 2 = [c1, c2] := rfl
  have hplace' : X86Env.place (List.eraseIdx [c1, c2, r] 2)
      (some (.at A)) (X86Env.pages (cap + cap)) = some (some A) := by
    rw [herase, hpages2]; exact hplace
  have hnextv : Kern.next k = (some (.at A), { k with oracle := rest }) := by
    simp [Kern.next, horacle]
  have hnext1 : (Kern.next k).1 = some (.at A) := by
    rw [hnextv]
  have hk1 : (Kern.next k).2 = { k with oracle := rest } := by
    rw [hnextv]
  simp only [X86Env.syscall, hfind, hbase, hlen, hrw]
  rw [if_neg (by simp), if_neg (by simp [hpages]),
    if_neg (by rw [hpages2]; omega : ¬ X86Env.pages (cap + cap) < cap)]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1 hk1
    simp only [] at hnext1 hk1
    subst hnext1
    subst hk1
    dsimp only []
    rw [hplace', herase]
    simp [hpages2, hlen]

/-- The growth block (12 steps from `base + 0xfc`): `mremap`
doubles the file buffer in place at `A`, consuming one oracle
placement; the state at the syscall (after 6 steps) is
oracle-legal.  The count and the buffer prefix are preserved. -/
theorem loader_grow {f : List Nat} {A sp0 cap count capB : Nat}
    {c1 c2 buf : Region} {s : X86Int.St} {rest : List Choice}
    (hrip : s.rip = base + 0xfc) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hmem : s.mem = [c1, c2, buf]) (hbufbase : buf.base = A)
    (hbufrw : buf.rw = true) (hbufhi : buf.hi = A + cap)
    (hdisj1 : c1.hi ≤ A ∨ A + capB ≤ c1.base)
    (hdisj2 : c2.hi ≤ A ∨ A + capB ≤ c2.base)
    (hcapB : cap + cap ≤ capB)
    (hcontent : s.mem.rdN A count = some (f.take count))
    (hfd : s.k.fds = [some ⟨f, count⟩])
    (horacle : s.k.oracle = .at A :: rest)
    (hcountcap : count ≤ cap)
    (hAmod : A % page = 0) (hApos : 0 < A)
    (hbound : A + (cap + cap) ≤ 2 ^ 47)
    (hcapmod : cap % page = 0) (hcappos : 0 < cap)
    (hcapW : cap + cap < W) (hA63 : A < 2 ^ 63) :
    ∃ s', run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0xd7 ∧
      s'.reg 3 = some A ∧ s'.reg 12 = some (cap + cap) ∧
      s'.reg 13 = some count ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 = some (sp0 + 24) ∧
      (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
      (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      s'.k.oracle = rest ∧ s'.k.fds = [some ⟨f, count⟩] ∧
      (∃ buf', s'.mem = [c1, c2, buf'] ∧ buf'.base = A ∧
        buf'.hi = A + (cap + cap) ∧ buf'.rw = true) ∧
      s'.mem.rdN A count = some (f.take count) ∧
      s'.regs.length = 16 ∧
      (∃ t, run stage1Im 6 (Out.ok s) = Out.ok t ∧ OracleLegalAt t) := by
  have hc := stage1_loaderCode
  obtain ⟨v8, hr8v⟩ := hr8
  obtain ⟨v9, hr9v⟩ := hr9
  have hdatalen : buf.data.length = cap := by
    have h := hbufhi
    unfold Region.hi at h
    omega
  have hpages : X86Env.pages cap = cap := pages_eq_of_mod_zero hcapmod
  have hpages2 : X86Env.pages (cap + cap) = cap + cap :=
    pages_eq_of_mod_zero (by simp [Nat.add_mod, hcapmod])
  have hfind : Mem.find [c1, c2, buf] A 0 = some (2, buf) := by
    have hd1 : c1.hi ≤ A ∨ A < c1.base := by
      rcases hdisj1 with h | h
      · exact Or.inl h
      · exact Or.inr (by omega)
    have hd2 : c2.hi ≤ A ∨ A < c2.base := by
      rcases hdisj2 with h | h
      · exact Or.inl h
      · exact Or.inr (by omega)
    exact find_third hd1 hd2 (by omega) (by rw [hbufhi]; omega)
  have hplace : X86Env.place [c1, c2] (some (.at A)) (cap + cap) =
      some (some A) := by
    have hd1 : c1.hi ≤ A ∨ A + (cap + cap) ≤ c1.base := by
      rcases hdisj1 with h | h
      · exact Or.inl h
      · exact Or.inr (by omega)
    have hd2 : c2.hi ≤ A ∨ A + (cap + cap) ≤ c2.base := by
      rcases hdisj2 with h | h
      · exact Or.inl h
      · exact Or.inr (by omega)
    exact place_at hAmod hApos hbound (free_pair hd1 hd2)
  have hsc : X86Env.syscall s.k s.mem 25 A cap (cap + cap)
        MREMAP_MAYMOVE v8 v9 =
      some (.ret A { s.k with oracle := rest }
        [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩]) := by
    rw [hmem]
    exact syscall_mremap_grow hfind hbufbase hdatalen hbufrw hpages
      hpages2 horacle hplace
  have hcontent' : Mem.rdN [c1, c2, buf] A count =
      some (f.take count) := by
    rwa [hmem] at hcontent
  have hrdN' : Mem.rdN [c1, c2,
        ⟨A, buf.data ++ List.replicate cap 0, true⟩] A count =
      some (f.take count) :=
    rdN_grow_preserve hdisj1 hdisj2 hbufbase (by omega) (by omega) hcontent'
  -- step 1: mov eax, 25
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some (25 % W)) with rip := s.rip + 5 } :=
    step_movI (by rw [hrip]; exact hc.dfc)
  obtain ⟨g1, hg1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some (25 % W)) with rip := s.rip + 5 } := ⟨_, rfl⟩
  have hg1rip : g1.rip = base + 0x101 := by
    rw [hg1]; show s.rip + 5 = base + 0x101; rw [hrip]
  have hlen1 : g1.regs.length = 16 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen16
  have hg1r0 : g1.reg 0 = some 25 := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg1r3 : g1.reg 3 = some A := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 3 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 0)]
    exact hr3
  have hg1r12 : g1.reg 12 = some cap := by
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 0)]
    exact hr12
  have hg1mem : g1.mem = s.mem := by rw [hg1]; rfl
  have hg1k : g1.k = s.k := by rw [hg1]; rfl
  -- step 2: mov rdi, rbx
  have e2 : X86Int.step stage1Im g1 =
      .ok { g1.set 7 (some A) with rip := g1.rip + 3 } :=
    step_mov (by rw [hg1rip]; exact hc.d101) hg1r3
  obtain ⟨g2, hg2⟩ : ∃ t : X86Int.St, t =
      { g1.set 7 (some A) with rip := g1.rip + 3 } := ⟨_, rfl⟩
  have hg2rip : g2.rip = base + 0x104 := by
    rw [hg2]; show g1.rip + 3 = base + 0x104; rw [hg1rip]
  have hlen2 : g2.regs.length = 16 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen1
  have hg2r7 : g2.reg 7 = some A := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg2r0 : g2.reg 0 = some 25 := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 7)]
    exact hg1r0
  have hg2r12 : g2.reg 12 = some cap := by
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 12 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 7)]
    exact hg1r12
  have hg2mem : g2.mem = s.mem := by rw [hg2]; exact hg1mem
  have hg2k : g2.k = s.k := by rw [hg2]; exact hg1k
  -- step 3: mov rsi, r12
  have e3 : X86Int.step stage1Im g2 =
      .ok { g2.set 6 (some cap) with rip := g2.rip + 3 } :=
    step_mov (by rw [hg2rip]; exact hc.d104) hg2r12
  obtain ⟨g3, hg3⟩ : ∃ t : X86Int.St, t =
      { g2.set 6 (some cap) with rip := g2.rip + 3 } := ⟨_, rfl⟩
  have hg3rip : g3.rip = base + 0x107 := by
    rw [hg3]; show g2.rip + 3 = base + 0x107; rw [hg2rip]
  have hlen3 : g3.regs.length = 16 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen2
  have hg3r6 : g3.reg 6 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg3r7 : g3.reg 7 = some A := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 6)]
    exact hg2r7
  have hg3r0 : g3.reg 0 = some 25 := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 6)]
    exact hg2r0
  have hg3r12 : g3.reg 12 = some cap := by
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 12 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (12 : Nat) ≠ 6)]
    exact hg2r12
  have hg3mem : g3.mem = s.mem := by rw [hg3]; exact hg2mem
  have hg3k : g3.k = s.k := by rw [hg3]; exact hg2k
  -- step 4: mov rdx, r12
  have e4 : X86Int.step stage1Im g3 =
      .ok { g3.set 2 (some cap) with rip := g3.rip + 3 } :=
    step_mov (by rw [hg3rip]; exact hc.d107) hg3r12
  obtain ⟨g4, hg4⟩ : ∃ t : X86Int.St, t =
      { g3.set 2 (some cap) with rip := g3.rip + 3 } := ⟨_, rfl⟩
  have hg4rip : g4.rip = base + 0x10a := by
    rw [hg4]; show g3.rip + 3 = base + 0x10a; rw [hg3rip]
  have hlen4 : g4.regs.length = 16 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen3
  have hg4r2 : g4.reg 2 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 2 = some cap
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg4r6 : g4.reg 6 = some cap := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 6 = some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 2)]
    exact hg3r6
  have hg4r7 : g4.reg 7 = some A := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 7 = some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 2)]
    exact hg3r7
  have hg4r0 : g4.reg 0 = some 25 := by
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 0 = some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 2)]
    exact hg3r0
  have hg4mem : g4.mem = s.mem := by rw [hg4]; exact hg3mem
  have hg4k : g4.k = s.k := by rw [hg4]; exact hg3k
  -- step 5: add rdx, rdx
  have hadd : (aluF .add cap cap).1 = cap + cap := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt hcapW]
  have e5 : X86Int.step stage1Im g4 =
      .ok { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } :=
    step_alu_add (by rw [hg4rip]; exact hc.d10a) hg4r2 hg4r2
  obtain ⟨g5, hg5⟩ : ∃ t : X86Int.St, t =
      { g4.set 2 (some (aluF .add cap cap).1) with
        fl := (aluF .add cap cap).2, rip := g4.rip + 3 } := ⟨_, rfl⟩
  have hg5rip : g5.rip = base + 0x10d := by
    rw [hg5]; show g4.rip + 3 = base + 0x10d; rw [hg4rip]
  have hlen5 : g5.regs.length = 16 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).regs.length = 16
    rw [len_set]; exact hlen4
  have hg5r2 : g5.reg 2 = some (cap + cap) := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 2 = some (cap + cap)
    rw [reg_set_self (by omega), hadd]
  have hg5r6 : g5.reg 6 = some cap := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 6 = some cap
    rw [reg_set_other (by decide : (6 : Nat) ≠ 2)]; exact hg4r6
  have hg5r7 : g5.reg 7 = some A := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 7 = some A
    rw [reg_set_other (by decide : (7 : Nat) ≠ 2)]; exact hg4r7
  have hg5r0 : g5.reg 0 = some 25 := by
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 0 = some 25
    rw [reg_set_other (by decide : (0 : Nat) ≠ 2)]; exact hg4r0
  have hg5mem : g5.mem = s.mem := by rw [hg5]; exact hg4mem
  have hg5k : g5.k = s.k := by rw [hg5]; exact hg4k
  -- step 6: mov r10, 1 (MREMAP_MAYMOVE)
  have e6 : X86Int.step stage1Im g5 =
      .ok { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } :=
    step_movI (by rw [hg5rip]; exact hc.d10d)
  obtain ⟨g6, hg6⟩ : ∃ t : X86Int.St, t =
      { g5.set 10 (some (1 % W)) with rip := g5.rip + 6 } := ⟨_, rfl⟩
  have hg6rip : g6.rip = base + 0x113 := by
    rw [hg6]; show g5.rip + 6 = base + 0x113; rw [hg5rip]
  have hlen6 : g6.regs.length = 16 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact hlen5
  have hg6r10 : g6.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 10 =
      some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_self (by omega)]
    decide
  have hg6r0 : g6.reg 0 = some 25 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 0 =
      some 25
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10)]
    exact hg5r0
  have hg6r7 : g6.reg 7 = some A := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 7 =
      some A
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 10)]
    exact hg5r7
  have hg6r6 : g6.reg 6 = some cap := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 6 =
      some cap
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 10)]
    exact hg5r6
  have hg6r2 : g6.reg 2 = some (cap + cap) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 2 =
      some (cap + cap)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 10)]
    exact hg5r2
  have hg6mem : g6.mem = s.mem := by rw [hg6]; exact hg5mem
  have hg6k : g6.k = s.k := by rw [hg6]; exact hg5k
  have hg6r8 : g6.reg 8 = some v8 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 8 =
      some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hr8v
  have hg6r9 : g6.reg 9 = some v9 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 9 =
      some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hr9v
  have hg6r13 : g6.reg 13 = some count := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hr13
  have hg6r14 : g6.reg 14 = some 3 := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 14 =
      some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hr14
  have hg6r4 : g6.reg 4 = some (sp0 + 24) := by
    rw [hg6]
    show ({ g5.set 10 (some (1 % W)) with rip := g5.rip + 6 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 10)]
    rw [hg5]
    show (g4.set 2 (some (aluF .add cap cap).1)).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg4]
    show ({ g3.set 2 (some cap) with rip := g3.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 2)]
    rw [hg3]
    show ({ g2.set 6 (some cap) with rip := g2.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6)]
    rw [hg2]
    show ({ g1.set 7 (some A) with rip := g1.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    rw [hg1]
    show ({ s.set 0 (some (25 % W)) with rip := s.rip + 5 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hr4
  -- legality at the syscall state, and the syscall itself
  have hsc6 : X86Env.syscall g6.k g6.mem 25 A cap (cap + cap)
        MREMAP_MAYMOVE v8 v9 =
      some (.ret A { s.k with oracle := rest }
        [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩]) := by
    rw [hg6k, hg6mem]; exact hsc
  have hlegal : OracleLegalAt g6 := by
    intro _hmem
    exact ⟨25, A, cap, cap + cap, MREMAP_MAYMOVE, v8, v9, A,
      { s.k with oracle := rest },
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩],
      hg6r0, hg6r7, hg6r6, hg6r2, hg6r10, hg6r8, hg6r9, hsc6⟩
  -- step 7: mremap(buf, cap, 2 * cap, MREMAP_MAYMOVE)
  have e7 : X86Int.step stage1Im g6 =
      .ok { ((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11 none with
        k := { s.k with oracle := rest },
        mem := [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩],
        rip := g6.rip + 2 } :=
    step_syscall_ret (by rw [hg6rip]; exact hc.d113) hg6r0 hg6r7 hg6r6
      hg6r2 hg6r10 hg6r8 hg6r9 hsc6
  obtain ⟨g7, hg7⟩ : ∃ t : X86Int.St, t =
      { ((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11 none with
        k := { s.k with oracle := rest },
        mem := [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩],
        rip := g6.rip + 2 } := ⟨_, rfl⟩
  have hg7rip : g7.rip = base + 0x115 := by
    rw [hg7]; show g6.rip + 2 = base + 0x115; rw [hg6rip]
  have hlen7 : g7.regs.length = 16 := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact hlen6
  have hg7r0 : g7.reg 0 = some A := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 0 = some A
    rw [reg_set_other (by decide : (0 : Nat) ≠ 11),
      reg_set_other (by decide : (0 : Nat) ≠ 1),
      reg_set_self (by omega)]
  have hg7r2 : g7.reg 2 = some (cap + cap) := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 2 = some (cap + cap)
    rw [reg_set_other (by decide : (2 : Nat) ≠ 11),
      reg_set_other (by decide : (2 : Nat) ≠ 1),
      reg_set_other (by decide : (2 : Nat) ≠ 0)]
    exact hg6r2
  have hg7r13 : g7.reg 13 = some count := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 13 = some count
    rw [reg_set_other (by decide : (13 : Nat) ≠ 11),
      reg_set_other (by decide : (13 : Nat) ≠ 1),
      reg_set_other (by decide : (13 : Nat) ≠ 0)]
    exact hg6r13
  have hg7r14 : g7.reg 14 = some 3 := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 14 = some 3
    rw [reg_set_other (by decide : (14 : Nat) ≠ 11),
      reg_set_other (by decide : (14 : Nat) ≠ 1),
      reg_set_other (by decide : (14 : Nat) ≠ 0)]
    exact hg6r14
  have hg7r4 : g7.reg 4 = some (sp0 + 24) := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 4 = some (sp0 + 24)
    rw [reg_set_other (by decide : (4 : Nat) ≠ 11),
      reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_other (by decide : (4 : Nat) ≠ 0)]
    exact hg6r4
  have hg7r10 : g7.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 10 = some MREMAP_MAYMOVE
    rw [reg_set_other (by decide : (10 : Nat) ≠ 11),
      reg_set_other (by decide : (10 : Nat) ≠ 1),
      reg_set_other (by decide : (10 : Nat) ≠ 0)]
    exact hg6r10
  have hg7r8 : g7.reg 8 = some v8 := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 8 = some v8
    rw [reg_set_other (by decide : (8 : Nat) ≠ 11),
      reg_set_other (by decide : (8 : Nat) ≠ 1),
      reg_set_other (by decide : (8 : Nat) ≠ 0)]
    exact hg6r8
  have hg7r9 : g7.reg 9 = some v9 := by
    rw [hg7]
    show (((g6.set 0 (some A)).set 1 (some (g6.rip + 2))).set 11
      none).reg 9 = some v9
    rw [reg_set_other (by decide : (9 : Nat) ≠ 11),
      reg_set_other (by decide : (9 : Nat) ≠ 1),
      reg_set_other (by decide : (9 : Nat) ≠ 0)]
    exact hg6r9
  have hg7mem : g7.mem =
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩] := by rw [hg7]
  have hg7k : g7.k = { s.k with oracle := rest } := by rw [hg7]
  -- step 8: test rax, rax
  have e8 : X86Int.step stage1Im g7 =
      .ok { g7 with fl := (aluF .and A A).2, rip := g7.rip + 3 } :=
    step_test (by rw [hg7rip]; exact hc.d115) hg7r0 hg7r0
  obtain ⟨g8, hg8⟩ : ∃ t : X86Int.St, t =
      { g7 with fl := (aluF .and A A).2, rip := g7.rip + 3 } := ⟨_, rfl⟩
  have hg8rip : g8.rip = base + 0x118 := by
    rw [hg8]; show g7.rip + 3 = base + 0x118; rw [hg7rip]
  have hg8sf : g8.fl.sf = some false := by
    rw [hg8]
    show (aluF .and A A).2.sf = some false
    have hland : Nat.land A A = A := by simp
    have hmsb : X86Int.msb A = false := by
      simp only [X86Int.msb]
      rw [Nat.div_eq_of_lt hA63]
      decide
    simp [aluF, hland, hmsb]
  have hg8r0 : g8.reg 0 = some A := by rw [hg8]; exact hg7r0
  have hg8r2 : g8.reg 2 = some (cap + cap) := by rw [hg8]; exact hg7r2
  have hg8r13 : g8.reg 13 = some count := by rw [hg8]; exact hg7r13
  have hg8r14 : g8.reg 14 = some 3 := by rw [hg8]; exact hg7r14
  have hg8r4 : g8.reg 4 = some (sp0 + 24) := by rw [hg8]; exact hg7r4
  have hg8r10 : g8.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg8]; exact hg7r10
  have hg8r8 : g8.reg 8 = some v8 := by rw [hg8]; exact hg7r8
  have hg8r9 : g8.reg 9 = some v9 := by rw [hg8]; exact hg7r9
  have hg8mem : g8.mem =
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg8]; exact hg7mem
  have hg8k : g8.k = { s.k with oracle := rest } := by
    rw [hg8]; exact hg7k
  have hlen8 : g8.regs.length = 16 := by rw [hg8]; exact hlen7
  -- step 9: js fail2 (not taken: A ≥ 0)
  have hg8fl8 : X86Int.cond g8.fl 8 = some false :=
    cond_cc8_of_sf_false hg8sf
  have e9 : X86Int.step stage1Im g8 =
      .ok { g8 with rip := g8.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := g8)
      (by rw [hg8rip]; exact hc.d118) hg8fl8
    simpa using h
  obtain ⟨g9, hg9⟩ : ∃ t : X86Int.St, t =
      { g8 with rip := g8.rip + 2 } := ⟨_, rfl⟩
  have hg9rip : g9.rip = base + 0x11a := by
    rw [hg9]; show g8.rip + 2 = base + 0x11a; rw [hg8rip]
  have hg9r0 : g9.reg 0 = some A := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 0 = some A
    rw [reg_with_rip]; exact hg8r0
  have hg9r2 : g9.reg 2 = some (cap + cap) := by
    rw [hg9]
    show ({ g8 with rip := g8.rip + 2 }).reg 2 = some (cap + cap)
    rw [reg_with_rip]; exact hg8r2
  have hg9r13 : g9.reg 13 = some count := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 13 = some count
    rw [reg_with_rip]; exact hg8r13
  have hg9r14 : g9.reg 14 = some 3 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 14 = some 3
    rw [reg_with_rip]; exact hg8r14
  have hg9r4 : g9.reg 4 = some (sp0 + 24) := by
    rw [hg9]
    show ({ g8 with rip := g8.rip + 2 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hg8r4
  have hg9r10 : g9.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg9]
    show ({ g8 with rip := g8.rip + 2 }).reg 10 = some MREMAP_MAYMOVE
    rw [reg_with_rip]; exact hg8r10
  have hg9r8 : g9.reg 8 = some v8 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 8 = some v8
    rw [reg_with_rip]; exact hg8r8
  have hg9r9 : g9.reg 9 = some v9 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).reg 9 = some v9
    rw [reg_with_rip]; exact hg8r9
  have hg9mem : g9.mem =
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg9]; exact hg8mem
  have hg9k : g9.k = { s.k with oracle := rest } := by
    rw [hg9]; exact hg8k
  have hlen9 : g9.regs.length = 16 := by
    rw [hg9]; show ({ g8 with rip := g8.rip + 2 }).regs.length = 16
    rw [len_with_rip]; exact hlen8
  -- step 10: mov rbx, rax
  have e10 : X86Int.step stage1Im g9 =
      .ok { g9.set 3 (some A) with rip := g9.rip + 3 } :=
    step_mov (by rw [hg9rip]; exact hc.d11a) hg9r0
  obtain ⟨g10, hg10⟩ : ∃ t : X86Int.St, t =
      { g9.set 3 (some A) with rip := g9.rip + 3 } := ⟨_, rfl⟩
  have hg10rip : g10.rip = base + 0x11d := by
    rw [hg10]; show g9.rip + 3 = base + 0x11d; rw [hg9rip]
  have hlen10 : g10.regs.length = 16 := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen9
  have hg10r3 : g10.reg 3 = some A := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg10r2 : g10.reg 2 = some (cap + cap) := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 2 =
      some (cap + cap)
    rw [reg_with_rip, reg_set_other (by decide : (2 : Nat) ≠ 3)]
    exact hg9r2
  have hg10r0 : g10.reg 0 = some A := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 0 = some A
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 3)]
    exact hg9r0
  have hg10r13 : g10.reg 13 = some count := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 13 =
      some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 3)]
    exact hg9r13
  have hg10r14 : g10.reg 14 = some 3 := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 3)]
    exact hg9r14
  have hg10r4 : g10.reg 4 = some (sp0 + 24) := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 4 =
      some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 3)]
    exact hg9r4
  have hg10r10 : g10.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 10 =
      some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 3)]
    exact hg9r10
  have hg10r8 : g10.reg 8 = some v8 := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 3)]
    exact hg9r8
  have hg10r9 : g10.reg 9 = some v9 := by
    rw [hg10]
    show ({ g9.set 3 (some A) with rip := g9.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 3)]
    exact hg9r9
  have hg10mem : g10.mem =
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg10]; exact hg9mem
  have hg10k : g10.k = { s.k with oracle := rest } := by
    rw [hg10]; exact hg9k
  -- step 11: mov r12, rdx
  have e11 : X86Int.step stage1Im g10 =
      .ok { g10.set 12 (some (cap + cap)) with rip := g10.rip + 3 } :=
    step_mov (by rw [hg10rip]; exact hc.d11d) hg10r2
  obtain ⟨g11, hg11⟩ : ∃ t : X86Int.St, t =
      { g10.set 12 (some (cap + cap)) with rip := g10.rip + 3 } :=
    ⟨_, rfl⟩
  have hg11rip : g11.rip = base + 0x120 := by
    rw [hg11]; show g10.rip + 3 = base + 0x120; rw [hg10rip]
  have hlen11 : g11.regs.length = 16 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen10
  have hg11r12 : g11.reg 12 = some (cap + cap) := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 12 = some (cap + cap)
    rw [reg_with_rip, reg_set_self (by omega)]
  have hg11r3 : g11.reg 3 = some A := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 3 = some A
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 12)]
    exact hg10r3
  have hg11r0 : g11.reg 0 = some A := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 0 = some A
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 12)]
    exact hg10r0
  have hg11r13 : g11.reg 13 = some count := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 13 = some count
    rw [reg_with_rip, reg_set_other (by decide : (13 : Nat) ≠ 12)]
    exact hg10r13
  have hg11r14 : g11.reg 14 = some 3 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 14 = some 3
    rw [reg_with_rip, reg_set_other (by decide : (14 : Nat) ≠ 12)]
    exact hg10r14
  have hg11r4 : g11.reg 4 = some (sp0 + 24) := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 12)]
    exact hg10r4
  have hg11r10 : g11.reg 10 = some MREMAP_MAYMOVE := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 10 = some MREMAP_MAYMOVE
    rw [reg_with_rip, reg_set_other (by decide : (10 : Nat) ≠ 12)]
    exact hg10r10
  have hg11r8 : g11.reg 8 = some v8 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 8 = some v8
    rw [reg_with_rip, reg_set_other (by decide : (8 : Nat) ≠ 12)]
    exact hg10r8
  have hg11r9 : g11.reg 9 = some v9 := by
    rw [hg11]
    show ({ g10.set 12 (some (cap + cap)) with
      rip := g10.rip + 3 }).reg 9 = some v9
    rw [reg_with_rip, reg_set_other (by decide : (9 : Nat) ≠ 12)]
    exact hg10r9
  have hg11mem : g11.mem =
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩] := by
    rw [hg11]; exact hg10mem
  have hg11k : g11.k = { s.k with oracle := rest } := by
    rw [hg11]; exact hg10k
  -- step 12: jmp readloop
  have e12 : X86Int.step stage1Im g11 =
      .ok { g11 with rip := base + 0xd7 } :=
    step_jmp (by rw [hg11rip]; exact hc.d120)
  obtain ⟨g12, hg12⟩ : ∃ t : X86Int.St, t =
      { g11 with rip := base + 0xd7 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok g1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hg1, run_zero]
  have r2 : run stage1Im 1 (Out.ok g1) = Out.ok g2 := by
    rw [run_succ_ok stage1Im 0 g1, e2, ← hg2, run_zero]
  have r3 : run stage1Im 1 (Out.ok g2) = Out.ok g3 := by
    rw [run_succ_ok stage1Im 0 g2, e3, ← hg3, run_zero]
  have r4 : run stage1Im 1 (Out.ok g3) = Out.ok g4 := by
    rw [run_succ_ok stage1Im 0 g3, e4, ← hg4, run_zero]
  have r5 : run stage1Im 1 (Out.ok g4) = Out.ok g5 := by
    rw [run_succ_ok stage1Im 0 g4, e5, ← hg5, run_zero]
  have r6 : run stage1Im 1 (Out.ok g5) = Out.ok g6 := by
    rw [run_succ_ok stage1Im 0 g5, e6, ← hg6, run_zero]
  have r7 : run stage1Im 1 (Out.ok g6) = Out.ok g7 := by
    rw [run_succ_ok stage1Im 0 g6, e7, ← hg7, run_zero]
  have r8 : run stage1Im 1 (Out.ok g7) = Out.ok g8 := by
    rw [run_succ_ok stage1Im 0 g7, e8, ← hg8, run_zero]
  have r9 : run stage1Im 1 (Out.ok g8) = Out.ok g9 := by
    rw [run_succ_ok stage1Im 0 g8, e9, ← hg9, run_zero]
  have r10 : run stage1Im 1 (Out.ok g9) = Out.ok g10 := by
    rw [run_succ_ok stage1Im 0 g9, e10, ← hg10, run_zero]
  have r11 : run stage1Im 1 (Out.ok g10) = Out.ok g11 := by
    rw [run_succ_ok stage1Im 0 g10, e11, ← hg11, run_zero]
  have r12 : run stage1Im 1 (Out.ok g11) = Out.ok g12 := by
    rw [run_succ_ok stage1Im 0 g11, e12, ← hg12, run_zero]
  have hrun6 : run stage1Im 6 (Out.ok s) = Out.ok g6 := by
    rw [show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r5, r6]
  have hrun : run stage1Im 12 (Out.ok s) = Out.ok g12 := by
    rw [show (12 : Nat) = 6 + 6 from rfl, run_add stage1Im 6 6, hrun6,
      show (6 : Nat) = 1 + 5 from rfl, run_add stage1Im 1 5, r7,
      show (5 : Nat) = 1 + 4 from rfl, run_add stage1Im 1 4, r8,
      show (4 : Nat) = 1 + 3 from rfl, run_add stage1Im 1 3, r9,
      show (3 : Nat) = 1 + 2 from rfl, run_add stage1Im 1 2, r10,
      show (2 : Nat) = 1 + 1 from rfl, run_add stage1Im 1 1, r11, r12]
  have hhi' : (⟨A, buf.data ++ List.replicate cap 0, true⟩ :
      Region).hi = A + (cap + cap) := by
    show A + (buf.data ++ List.replicate cap 0).length = A + (cap + cap)
    rw [List.length_append, List.length_replicate, hdatalen]
  refine ⟨g12, hrun, by rw [hg12], ?_, ?_, ?_, ?_, ?_, ⟨A, ?_⟩,
    ⟨MREMAP_MAYMOVE, ?_⟩, ⟨v8, ?_⟩, ⟨v9, ?_⟩, ?_, ?_,
    ⟨⟨A, buf.data ++ List.replicate cap 0, true⟩, ?_, rfl, hhi', rfl⟩,
    ?_, ?_, ⟨g6, hrun6, hlegal⟩⟩
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 3 = some A
    rw [reg_with_rip]; exact hg11r3
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 12 = some (cap + cap)
    rw [reg_with_rip]; exact hg11r12
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 13 = some count
    rw [reg_with_rip]; exact hg11r13
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 14 = some 3
    rw [reg_with_rip]; exact hg11r14
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 4 = some (sp0 + 24)
    rw [reg_with_rip]; exact hg11r4
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 0 = some A
    rw [reg_with_rip]; exact hg11r0
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 10 = some MREMAP_MAYMOVE
    rw [reg_with_rip]; exact hg11r10
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 8 = some v8
    rw [reg_with_rip]; exact hg11r8
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).reg 9 = some v9
    rw [reg_with_rip]; exact hg11r9
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).k.oracle = rest
    rw [hg11k]
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).k.fds = [some ⟨f, count⟩]
    rw [hg11k]
    exact hfd
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).mem =
      [c1, c2, ⟨A, buf.data ++ List.replicate cap 0, true⟩]
    exact hg11mem
  · rw [hg12]
    show Mem.rdN ({ g11 with rip := base + 0xd7 }).mem A count =
      some (f.take count)
    rw [hg11mem]
    exact hrdN'
  · rw [hg12]
    show ({ g11 with rip := base + 0xd7 }).regs.length = 16
    rw [len_with_rip]; exact hlen11

/-- The whole read loop, by strong induction on the bytes
remaining: every iteration moves at least one byte, growths
consume one replicated placement each, and the capacity bound
`f.length < cap * 2 ^ g` is preserved (with `A + cap * 2 ^ g`
inside the address bound).  The tail after the replicated
placements begins with a placement, which is all the reads need
once the growth entries run out (reads do not consume it).  Ends
just after the jump at `loaded`, at the parser entry
`base + 0x134`, with the whole file in the buffer. -/
theorem loader_read_loop : ∀ rem : Nat,
    ∀ {f : List Nat} {A sp0 cap count g capB : Nat}
      {c1 c2 buf : Region} {s : X86Int.St} {tail : List Choice},
      f.length - count = rem →
      s.rip = base + 0xd7 → s.regs.length = 16 →
      s.reg 3 = some A → s.reg 12 = some cap →
      s.reg 13 = some count → s.reg 14 = some 3 →
      s.reg 4 = some (sp0 + 24) →
      (∃ v, s.reg 0 = some v) → (∃ v, s.reg 10 = some v) →
      (∃ v, s.reg 8 = some v) → (∃ v, s.reg 9 = some v) →
      s.mem = [c1, c2, buf] → buf.base = A → buf.rw = true →
      buf.hi = A + cap → (c1.hi ≤ A ∨ A + capB ≤ c1.base) →
      (c2.hi ≤ A ∨ A + capB ≤ c2.base) → cap * 2 ^ g ≤ capB →
      s.mem.rdN A count = some (f.take count) →
      s.k.fds = [some ⟨f, count⟩] →
      s.k.oracle = List.replicate g (.at A) ++ tail →
      (∃ B, tail.head? = some (.at B)) →
      (∀ b ∈ f, b < 256) →
      count ≤ f.length → count < cap →
      A % page = 0 → 0 < A → A < 2 ^ 63 →
      cap % page = 0 → 0 < cap →
      A + cap * 2 ^ g ≤ 2 ^ 47 → f.length < cap * 2 ^ g →
      (g = 0 ∨ cap * 2 ^ (g - 1) ≤ f.length) →
      ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
        s'.rip = base + 0x134 ∧
        s'.reg 3 = some A ∧ (∃ cap', s'.reg 12 = some cap') ∧
        s'.reg 13 = some f.length ∧ s'.reg 14 = some 3 ∧
        s'.reg 4 = some (sp0 + 24) ∧
        (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
        (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
        s'.k.oracle = tail ∧
        s'.k.fds = [some ⟨f, f.length⟩] ∧
        (∃ buf', s'.mem = [c1, c2, buf'] ∧ buf'.base = A ∧
          (∃ cap', buf'.hi = A + cap' ∧ cap' ≤ capB) ∧
          buf'.rw = true) ∧
        s'.mem.rdN A f.length = some f ∧
        s'.regs.length = 16 := by
  intro rem
  induction rem using Nat.strongRecOn with
  | _ rem ih =>
  intro f A sp0 cap count g capB c1 c2 buf s tail hrem hrip hlen16 hr3
    hr12 hr13 hr14 hr4 hr0 hr10 hr8 hr9 hmem hbufbase hbufrw hbufhi
    hdisj1 hdisj2 hcapB hcontent hfd horacle htail hfB hcount hcountcap
    hAmod hApos hA63 hcapmod hcappos hbound hflen hgmin
  have hc := stage1_loaderCode
  have hcaple : cap ≤ cap * 2 ^ g := by
    cases g with
    | zero => simp
    | succ g =>
      have h1 : 1 ≤ 2 ^ (g + 1) := Nat.one_le_two_pow
      calc cap = cap * 1 := by rw [Nat.mul_one]
        _ ≤ cap * 2 ^ (g + 1) := Nat.mul_le_mul_left cap h1
  have hW : W = 2 ^ 64 := rfl
  have hAcap : A + cap < W := by omega
  have hcap63 : cap < 2 ^ 63 := by omega
  have hhead : ∃ B, s.k.oracle.head? = some (.at B) := by
    cases g with
    | zero =>
      obtain ⟨B, hB⟩ := htail
      refine ⟨B, ?_⟩
      rw [horacle]
      simpa using hB
    | succ g' =>
      refine ⟨A, ?_⟩
      rw [horacle, List.replicate_succ]
      simp
  obtain ⟨s1, hrun1, hor1, hs1r0, hs1r3, hs1r12, hs1r13, hs1r14, hs1r4,
    hs1r10, hs1r8, hs1r9, hs1oracle, hs1fds,
    ⟨buf1, hs1mem, hbuf1base, hbuf1hi, hbuf1rw⟩, hs1content, hs1len⟩ :=
    loader_read_once hrip hlen16 hr3 hr12 hr13 hr14 hr4 hr0 hr10 hr8 hr9
      hmem hbufbase hbufrw hbufhi hdisj1 hdisj2 (by omega) hcontent hfd
      hhead hfB hcount hcountcap hAcap hcap63
  obtain ⟨n, hn⟩ : ∃ m : Nat,
      m = min (cap - count) (f.length - count) := ⟨_, rfl⟩
  rw [← hn] at hs1r0 hs1fds hs1content
  have hncap : count + n ≤ cap := by
    have h1 : n ≤ cap - count := by
      rw [hn]; exact Nat.min_le_left _ _
    omega
  have hnlen : count + n ≤ f.length := by
    have h1 : n ≤ f.length - count := by
      rw [hn]; exact Nat.min_le_right _ _
    omega
  rcases hor1 with ⟨hs1rip, hlen_eq⟩ | ⟨hs1rip, hlt⟩
  · -- EOF: jump from loaded (0x122) to the parser entry (0x134)
    have hn0 : n = 0 := by
      rw [hn]
      simp [show f.length - count = 0 by omega]
    have e11 : X86Int.step stage1Im s1 =
        .ok { s1 with rip := base + 0x134 } :=
      step_jmp (by rw [hs1rip]; exact hc.d122)
    obtain ⟨s2, hs2⟩ : ∃ t : X86Int.St, t =
        { s1 with rip := base + 0x134 } := ⟨_, rfl⟩
    have r11 : run stage1Im 1 (Out.ok s1) = Out.ok s2 := by
      rw [run_succ_ok stage1Im 0 s1, e11, ← hs2, run_zero]
    have hrun : run stage1Im 11 (Out.ok s) = Out.ok s2 := by
      rw [show (11 : Nat) = 10 + 1 from rfl, run_add stage1Im 10 1,
        hrun1, r11]
    have hcontent2 : s2.mem.rdN A f.length = some f := by
      rw [hn0, Nat.add_zero] at hs1content
      rw [hs2]
      show Mem.rdN ({ s1 with rip := base + 0x134 }).mem A f.length =
        some f
      have htf : f.take f.length = f := by simp
      rw [← hlen_eq, htf] at hs1content
      exact hs1content
    have hg0 : g = 0 := by
      by_cases hg0 : g = 0
      · exact hg0
      · have hge : cap * 2 ^ (g - 1) ≤ f.length := by
          rcases hgmin with h | h
          · omega
          · exact h
        have h1 : cap ≤ cap * 2 ^ (g - 1) := by
          calc cap = cap * 1 := by rw [Nat.mul_one]
            _ ≤ cap * 2 ^ (g - 1) :=
              Nat.mul_le_mul_left cap Nat.one_le_two_pow
        omega
    refine ⟨11, s2, hrun, by rw [hs2], ?_, ⟨cap, ?_⟩, ?_, ?_, ?_,
      ⟨n, ?_⟩, ?_, ?_, ?_, ?_, ?_,
      ⟨buf1, ?_, hbuf1base, ⟨cap, hbuf1hi, by omega⟩, hbuf1rw⟩,
      hcontent2, ?_⟩
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 3 = some A
      rw [reg_with_rip]; exact hs1r3
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 12 = some cap
      rw [reg_with_rip]; exact hs1r12
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 13 = some f.length
      rw [reg_with_rip, hlen_eq]; exact hs1r13
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 14 = some 3
      rw [reg_with_rip]; exact hs1r14
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 4 = some (sp0 + 24)
      rw [reg_with_rip]; exact hs1r4
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 0 = some n
      rw [reg_with_rip]; exact hs1r0
    · obtain ⟨v, hv⟩ := hs1r10
      refine ⟨v, ?_⟩
      rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 10 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hs1r8
      refine ⟨v, ?_⟩
      rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 8 = some v
      rw [reg_with_rip]; exact hv
    · obtain ⟨v, hv⟩ := hs1r9
      refine ⟨v, ?_⟩
      rw [hs2]
      show ({ s1 with rip := base + 0x134 }).reg 9 = some v
      rw [reg_with_rip]; exact hv
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).k.oracle = tail
      rw [hs1oracle, horacle, hg0]
      simp
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).k.fds = [some ⟨f, f.length⟩]
      rw [hs1fds, hn0, Nat.add_zero, ← hlen_eq]
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).mem = [c1, c2, buf1]
      exact hs1mem
    · rw [hs2]
      show ({ s1 with rip := base + 0x134 }).regs.length = 16
      rw [len_with_rip]; exact hs1len
  · -- a positive read: update the count, then loop or grow
    have hn1 : 0 < n := Nat.pos_of_ne_zero (by
      intro hn0
      rw [hn] at hn0
      omega)
    have hrem2 : f.length - (count + n) < rem := by omega
    obtain ⟨s2, hrun2, hor2, hs2r13, hs2r3, hs2r12, hs2r14, hs2r4,
      hs2r0, hs2r10, hs2r8, hs2r9, hs2mem, hs2k, hs2len⟩ :=
      loader_count_update hs1rip hs1len hs1r0 hs1r3 hs1r12 hs1r13
        hs1r14 hs1r4 hs1r10 hs1r8 hs1r9 hncap (by omega)
    have hs2content : s2.mem.rdN A (count + n) =
        some (f.take (count + n)) := by
      rwa [hs2mem]
    have hs2fds : s2.k.fds = [some ⟨f, count + n⟩] := by
      rw [hs2k]; exact hs1fds
    have hs2oracle : s2.k.oracle = List.replicate g (.at A) ++ tail := by
      rw [hs2k, hs1oracle, horacle]
    have hs2mem' : s2.mem = [c1, c2, buf1] := by
      rw [hs2mem]; exact hs1mem
    rcases hor2 with ⟨hs2rip, hltcap⟩ | ⟨hs2rip, hfull⟩
    · obtain ⟨k, s3, hrun3, hrip3, hr33, hcap3, hr133, hr143, hr43,
        hr03, hr103, hr83, hr93, hor3, hfd3, hmem3, hcontent3,
        hlen3⟩ :=
        ih (f.length - (count + n)) hrem2 rfl hs2rip hs2len hs2r3
          hs2r12 hs2r13 hs2r14 hs2r4 hs2r0 hs2r10 hs2r8 hs2r9 hs2mem'
          hbuf1base hbuf1rw hbuf1hi hdisj1 hdisj2 hcapB hs2content hs2fds
          hs2oracle htail hfB hnlen hltcap hAmod hApos hA63 hcapmod
          hcappos hbound hflen hgmin
      have hrun : run stage1Im (14 + k) (Out.ok s) = Out.ok s3 := by
        rw [show 14 + k = 10 + (4 + k) by omega, run_add stage1Im 10 _,
          hrun1, show 4 + k = 4 + k from rfl, run_add stage1Im 4 k,
          hrun2, hrun3]
      exact ⟨14 + k, s3, hrun, hrip3, hr33, hcap3, hr133, hr143, hr43,
        hr03, hr103, hr83, hr93, hor3, hfd3, hmem3, hcontent3, hlen3⟩
    · -- the buffer is exactly full: grow, then loop
      have hg1 : 1 ≤ g := by
        by_cases hg0 : g = 0
        · rw [hg0] at hflen
          simp at hflen
          omega
        · omega
      obtain ⟨g', hg'⟩ : ∃ g', g = g' + 1 := ⟨g - 1, by omega⟩
      have hor2' : s2.k.oracle =
          .at A :: (List.replicate g' (.at A) ++ tail) := by
        rw [hs2oracle, hg', List.replicate_succ, List.cons_append]
      have hbound2 : A + (cap + cap) ≤ 2 ^ 47 := by
        have h1 : cap + cap ≤ cap * 2 ^ g := by
          rw [hg']
          have h2 : (2 : Nat) ≤ 2 ^ (g' + 1) := by
            calc (2 : Nat) = 2 ^ 1 := by decide
              _ ≤ 2 ^ (g' + 1) :=
                Nat.pow_le_pow_right (by decide) (by omega)
          calc cap + cap = cap * 2 := by omega
            _ ≤ cap * 2 ^ (g' + 1) := Nat.mul_le_mul_left cap h2
        omega
      obtain ⟨s3, hrun3, hs3rip, hs3r3, hs3r12, hs3r13, hs3r14, hs3r4,
        hs3r0, hs3r10, hs3r8, hs3r9, hs3oracle, hs3fds,
        ⟨buf3, hs3mem, hbuf3base, hbuf3hi, hbuf3rw⟩, hs3content,
        hs3len, -⟩ :=
        loader_grow hs2rip hs2len hs2r3 hs2r12 hs2r13 hs2r14 hs2r4
          hs2r8 hs2r9 hs2mem' hbuf1base hbuf1rw hbuf1hi hdisj1 hdisj2
          (by
            have hcc : cap + cap ≤ cap * 2 ^ g := by
              rw [hg']
              have h2 : (2 : Nat) ≤ 2 ^ (g' + 1) := by
                calc (2 : Nat) = 2 ^ 1 := by decide
                  _ ≤ 2 ^ (g' + 1) :=
                    Nat.pow_le_pow_right (by decide) (by omega)
              calc cap + cap = cap * 2 := by omega
                _ ≤ cap * 2 ^ (g' + 1) := Nat.mul_le_mul_left cap h2
            omega)
          hs2content hs2fds hor2' (by omega) hAmod hApos hbound2
          hcapmod hcappos (by omega) hA63
      have hcapmod2 : (cap + cap) % page = 0 := by
        simp [Nat.add_mod, hcapmod]
      have hpow : (2 : Nat) * 2 ^ g' = 2 ^ (g' + 1) := by
        rw [Nat.pow_succ, Nat.mul_comm]
      have hcapgrow : (cap + cap) * 2 ^ g' = cap * 2 ^ g := by
        rw [hg', ← hpow, show cap + cap = cap * 2 by omega,
          Nat.mul_assoc]
      have hbound3 : A + (cap + cap) * 2 ^ g' ≤ 2 ^ 47 := by omega
      have hflen3 : f.length < (cap + cap) * 2 ^ g' := by omega
      obtain ⟨k, s4, hrun4, hrip4, hr34, hcap4, hr134, hr144, hr44,
          hr04, hr104, hr84, hr94, hor4, hfd4, hmem4, hcontent4,
          hlen4⟩ :=
        ih (f.length - (count + n)) hrem2 rfl hs3rip hs3len hs3r3
          hs3r12 hs3r13 hs3r14 hs3r4 hs3r0 hs3r10 hs3r8 hs3r9 hs3mem
          hbuf3base hbuf3rw hbuf3hi hdisj1 hdisj2
          (by rwa [hcapgrow]) hs3content hs3fds
          hs3oracle htail hfB hnlen (by omega) hAmod hApos hA63
          hcapmod2 (by omega) hbound3 hflen3 (by
            by_cases hg'0 : g' = 0
            · exact Or.inl hg'0
            · refine Or.inr ?_
              have hge : cap * 2 ^ (g - 1) ≤ f.length := by
                rcases hgmin with h | h
                · omega
                · exact h
              have e1 : g - 1 = g' := by omega
              have e2 : (cap + cap) * 2 ^ (g' - 1) = cap * 2 ^ g' := by
                obtain ⟨g'', hg''⟩ : ∃ m, g' = m + 1 := ⟨g' - 1, by omega⟩
                rw [hg'']
                simp only [Nat.add_sub_cancel]
                calc (cap + cap) * 2 ^ g'' = cap * (2 * 2 ^ g'') := by
                      rw [show cap + cap = cap * 2 by omega,
                        Nat.mul_assoc]
                  _ = cap * 2 ^ (g'' + 1) := by
                      rw [Nat.pow_succ, Nat.mul_comm (2 ^ g'') 2]
              rw [e2, ← e1]
              exact hge)
      have hrun : run stage1Im (26 + k) (Out.ok s) = Out.ok s4 := by
        rw [show 26 + k = 10 + (16 + k) by omega, run_add stage1Im 10 _,
          hrun1, show 16 + k = 4 + (12 + k) by omega,
          run_add stage1Im 4 _, hrun2, show 12 + k = 12 + k from rfl,
          run_add stage1Im 12 k, hrun3, hrun4]
      exact ⟨26 + k, s4, hrun, hrip4, hr34, hcap4, hr134, hr144, hr44,
        hr04, hr104, hr84, hr94, hor4, hfd4, hmem4, hcontent4, hlen4⟩

/-- The growth budget is exact: one fewer growth would not
hold the file. -/
theorem growths_min (L : Nat) :
    stage1Growths L = 0 ∨
      65536 * 2 ^ (stage1Growths L - 1) ≤ L := by
  unfold stage1Growths
  by_cases hL : L < 65536
  · rw [if_pos hL]
    exact Or.inl rfl
  · rw [if_neg hL]
    refine Or.inr ?_
    have hq : 0 < L / 65536 := by
      rw [Nat.div_pos_iff]
      omega
    have hle : 2 ^ (Nat.log2 (L / 65536)) ≤ L / 65536 := by
      by_cases h : L / 65536 < 2 ^ (Nat.log2 (L / 65536))
      · have h2 := (Nat.log2_lt (show L / 65536 ≠ 0 by omega)).2 h
        omega
      · omega
    have h1 : 65536 * 2 ^ (Nat.log2 (L / 65536)) ≤
        65536 * (L / 65536) := Nat.mul_le_mul_left 65536 hle
    have h2 : 65536 * (L / 65536) ≤ L := Nat.mul_div_le L 65536
    have e : Nat.log2 (L / 65536) + 1 - 1 = Nat.log2 (L / 65536) := by
      omega
    rw [e]
    omega

theorem growths_cap_gt (L : Nat) :
    L < 65536 * 2 ^ (stage1Growths L) := by
  unfold stage1Growths
  by_cases hL : L < 65536
  · rw [if_pos hL]
    simp [hL]
  · rw [if_neg hL]
    have hq : 0 < L / 65536 := by
      rw [Nat.div_pos_iff]
      omega
    have hlt : L / 65536 < 2 ^ (Nat.log2 (L / 65536) + 1) :=
      (Nat.log2_lt (by omega)).1 (Nat.lt_succ_self _)
    have hdiv := Nat.div_add_mod L 65536
    have hmod := Nat.mod_lt L (by decide : 0 < 65536)
    have h1 : L < 65536 * (L / 65536 + 1) := by omega
    have h2 : 65536 * (L / 65536 + 1) ≤
        65536 * 2 ^ (Nat.log2 (L / 65536) + 1) :=
      Nat.mul_le_mul_left 65536 hlt
    omega

theorem growths_cap_le_arena (f : List Nat) :
    65536 * 2 ^ (stage1Growths f.length) ≤ arenaFileBytes f := by
  have hge : 65536 + 2 * f.length ≤ arenaFileBytes f := by
    unfold arenaFileBytes
    exact pages_ge _
  unfold stage1Growths
  by_cases hL : f.length < 65536
  · rw [if_pos hL]
    simp
    omega
  · rw [if_neg hL]
    have hq : 0 < f.length / 65536 := by
      rw [Nat.div_pos_iff]
      omega
    have hle : 2 ^ (Nat.log2 (f.length / 65536)) ≤ f.length / 65536 := by
      by_cases h : f.length / 65536 < 2 ^ (Nat.log2 (f.length / 65536))
      · have := (Nat.log2_lt (by omega :
          f.length / 65536 ≠ 0)).2 h
        omega
      · omega
    have hdm : f.length / 65536 * 65536 ≤ f.length :=
      Nat.div_mul_le_self _ _
    have h1 : 65536 * 2 ^ (Nat.log2 (f.length / 65536) + 1) ≤
        65536 + 2 * f.length := by
      rw [Nat.pow_succ]
      omega
    omega

/-- A register other than `r11` (the one register a syscall
clears) that holds a value keeps holding a value after any
single step, and hence after any run: every other write is a
`some`. -/
theorem exec_reg_some_ne11 {ins : Ins} {len : Nat} {s s' : X86Int.St}
    {i v : Nat} (hi : i ≠ 11) (hv : s.reg i = some v)
    (h : exec ins len s = .ok s') : ∃ v', s'.reg i = some v' := by
  cases ins <;> simp only [exec] at h
  all_goals
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
  all_goals
    simp only [St.reg, St.set, List.getElem?_set]
    <;> (try split) <;> (try split) <;> (try split) <;> (try split)
    <;> (try split)
    <;> (first | simp_all [St.reg] | (exfalso; omega) | simp_all)

theorem step_reg_some_ne11 {im : Nat → Option (Ins × Nat)}
    {s s' : X86Int.St} {i v : Nat} (hi : i ≠ 11)
    (hv : s.reg i = some v) (h : X86Int.step im s = .ok s') :
    ∃ v', s'.reg i = some v' := by
  unfold X86Int.step at h
  cases him : im s.rip with
  | none => rw [him] at h; cases h
  | some p =>
    obtain ⟨ins, len⟩ := p
    rw [him] at h
    exact exec_reg_some_ne11 hi hv h

theorem run_reg_some_ne11 {im : Nat → Option (Ins × Nat)} :
    ∀ (n : Nat) {s s' : X86Int.St} {i v : Nat}, i ≠ 11 →
      s.reg i = some v → run im n (Out.ok s) = Out.ok s' →
      ∃ v', s'.reg i = some v' := by
  intro n
  induction n with
  | zero =>
    intro s s' i v hi hv h
    simp only [X86Int.run] at h
    rw [← Out.ok.inj h]
    exact ⟨v, hv⟩
  | succ n ih =>
    intro s s' i v hi hv h
    have h1 : run im (n + 1) (Out.ok s) = run im n (X86Int.step im s) :=
      rfl
    rw [h1] at h
    cases hs : X86Int.step im s with
    | ok t =>
      rw [hs] at h
      obtain ⟨v', hv'⟩ := step_reg_some_ne11 hi hv hs
      exact ih hi hv' h
    | exit c =>
      rw [hs] at h
      cases n <;> simp [X86Int.run] at h
    | stuck =>
      rw [hs] at h
      cases n <;> simp [X86Int.run] at h

/-- A byte write succeeds exactly when `find` hits a writable
region (the written value is irrelevant). -/
theorem wr_isSome_iff {m : Mem} {a u : Nat} :
    (m.wr a u).isSome ↔
      ∃ i r, Mem.find m a 0 = some (i, r) ∧ r.rw = true := by
  constructor
  · intro h
    cases hf : Mem.find m a 0 with
    | none => simp [Mem.wr, hf] at h
    | some p =>
      obtain ⟨i, r⟩ := p
      refine ⟨i, r, rfl, ?_⟩
      by_cases hrw : r.rw = true
      · exact hrw
      · simp [Mem.wr, hf, hrw] at h
  · rintro ⟨i, r, hf, hrw⟩
    simp [Mem.wr, hf, hrw]

/-- `find` in an append whose last region is the first hit. -/
theorem find_append_last : ∀ (m : Mem) (k : Nat) {r : Region}
    {a : Nat},
    (∀ r' ∈ m, ¬ (r'.base ≤ a ∧ a < r'.hi)) →
    r.base ≤ a ∧ a < r.hi →
    Mem.find (m ++ [r]) a k = some (k + m.length, r) := by
  intro m
  induction m with
  | nil =>
    intro k r a _ hr
    show Mem.find [r] a k = some (k, r)
    unfold Mem.find
    rw [if_pos hr]
  | cons r0 rest ih =>
    intro k r a hm hr
    have h0 : ¬ (r0.base ≤ a ∧ a < r0.hi) := hm r0 (.head _)
    have hg : Mem.find ((r0 :: rest) ++ [r]) a k =
        (if r0.base ≤ a ∧ a < r0.hi then some (k, r0)
          else Mem.find (rest ++ [r]) a (k + 1)) := rfl
    have hih := ih (k + 1) (fun r' hr' => hm r' (.tail _ hr')) hr
    have eidx : k + (r0 :: rest).length = k + 1 + rest.length := by
      simp only [List.length_cons]; omega
    rw [hg, if_neg h0, hih, eidx]

/-- One step of a successful `wrN`. -/
theorem wrN_cons_isSome {m : Mem} {a b : Nat} {bs : List Nat}
    (h : (m.wrN a (b :: bs)).isSome) :
    ∃ m1, m.wr a b = some m1 ∧ (m1.wrN (a + 1) bs).isSome := by
  have h2 : ((m.wr a b).bind (fun mm => mm.wrN (a + 1) bs)).isSome := h
  obtain ⟨m', hm'⟩ := Option.isSome_iff_exists.mp h2
  rw [Option.bind_eq_some_iff] at hm'
  obtain ⟨m1, h1, h3⟩ := hm'
  exact ⟨m1, h1, Option.isSome_iff_exists.mpr ⟨m', h3⟩⟩

/-- From a successful `wrN`, every byte offset is writable with
any value. -/
theorem wrN_isSome_decomp : ∀ (bs : List Nat) {m : Mem} {a : Nat},
    (m.wrN a bs).isSome → ∀ (j : Nat), j < bs.length → ∀ u,
      (m.wr (a + j) u).isSome := by
  intro bs
  induction bs with
  | nil => intro m a _ j hj _; simp at hj
  | cons b bs ih =>
    intro m a h j hj u
    obtain ⟨m1, h1, hrest⟩ := wrN_cons_isSome h
    cases j with
    | zero =>
      have hh : (m.wr a b).isSome :=
        Option.isSome_iff_exists.mpr ⟨m1, h1⟩
      have hf := wr_isSome_iff.mp hh
      exact wr_isSome_iff.mpr hf
    | succ j =>
      have hj' : j < bs.length := by
        simp only [List.length_cons] at hj; omega
      have hh := ih hrest j hj' u
      have hsh : wshape m = wshape m1 := (wr_wshape h1).symm
      have hw := wr_some_of_wshape hsh hh
      rwa [show a + 1 + j = a + (j + 1) by omega] at hw

/-- Assemble a `wrN` over a same-shape memory from per-byte
writability in the original. -/
theorem wrN_isSome_of_forall_wr : ∀ (bs : List Nat) {m m' : Mem}
    {a : Nat},
    wshape m' = wshape m →
    (∀ (j : Nat), j < bs.length → ∀ u, (m.wr (a + j) u).isSome) →
    (m'.wrN a bs).isSome := by
  intro bs
  induction bs with
  | nil =>
    intro m m' a _ _
    exact Option.isSome_iff_exists.mpr ⟨m', rfl⟩
  | cons b bs ih =>
    intro m m' a hsh h
    have hw : (m'.wr a b).isSome :=
      wr_some_of_wshape hsh (h 0 (by simp) b)
    obtain ⟨m'', hm''⟩ := Option.isSome_iff_exists.mp hw
    have h2 : ((m'.wr a b).bind (fun mm => mm.wrN (a + 1) bs)).isSome := by
      rw [hm'']
      show (m''.wrN (a + 1) bs).isSome
      have hsh2 : wshape m'' = wshape m := (wr_wshape hm'').trans hsh
      refine ih hsh2 (fun j hj u => ?_)
      have hh := h (j + 1) (by simp only [List.length_cons]; omega) u
      rwa [show a + 1 + j = a + (j + 1) by omega]
    exact h2

theorem bytes8_length (v : Nat) : (bytes8 v).length = 8 := by
  simp [bytes8]

/-- A `wr64` from per-byte writability. -/
theorem wr64_isSome_of_forall_wr {m : Mem} {a v : Nat}
    (h : ∀ (j : Nat), j < 8 → ∀ u, (m.wr (a + j) u).isSome) :
    (wr64 m a v).isSome := by
  show (m.wrN a (bytes8 v)).isSome
  exact wrN_isSome_of_forall_wr (bs := bytes8 v) rfl
    (fun j hj u => h j (by rw [bytes8_length] at hj; exact hj) u)

/-- From a successful `wr64`, every byte offset is writable. -/
theorem wr_isSome_of_wr64 {m : Mem} {a v : Nat}
    (h : (wr64 m a v).isSome) {j : Nat} (hj : j < 8) (u : Nat) :
    (m.wr (a + j) u).isSome := by
  have h2 : (m.wrN a (bytes8 v)).isSome := h
  exact wrN_isSome_decomp _ h2 j (by rw [bytes8_length]; exact hj) u

/-- `StackRoom` is monotone in the depth. -/
theorem stackRoom_mono {s : X86Int.St} {k k' : Nat}
    (h : StackRoom s k) (hle : k' ≤ k) : StackRoom s k' := by
  intro sp hr4 i hi v
  exact h sp hr4 i (by omega) v

/-- `StackRoom` transports to a same-shape memory with the
stack pointer lowered by whole slots. -/
theorem stackRoom_transport {s s' : X86Int.St}
    {k k' t sp : Nat}
    (hsh : wshape s'.mem = wshape s.mem)
    (hr4 : s.reg 4 = some sp) (hr4' : s'.reg 4 = some (sp - 8 * t))
    (hroom : StackRoom s k) (hk : t + k' ≤ k)
    (hge : 8 * k ≤ sp) (hspW : sp < W) :
    StackRoom s' k' := by
  intro sp'' h4'' i hi v
  have hsp'' : sp'' = sp - 8 * t := by
    rw [hr4'] at h4''
    exact (Option.some.inj h4'').symm
  have hmod : ∀ d : Nat, d ≤ sp → (sp + W - d) % W = sp - d := by
    intro d hd
    have hh : sp + W - d = (sp - d) + W := by omega
    rw [hh, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hA : (sp'' + W - 8 * (i + 1)) % W = sp - 8 * (t + i + 1) := by
    have ee : sp - 8 * t + W - 8 * (i + 1) =
        sp + W - 8 * (t + i + 1) := by omega
    rw [hsp'', ee, hmod (8 * (t + i + 1)) (by omega)]
  have hw64 := hroom sp hr4 (t + i) (by omega) v
  rw [hmod (8 * (t + i + 1)) (by omega)] at hw64
  rw [hA]
  exact wr64_isSome_of_forall_wr (fun j hj u =>
    wr_some_of_wshape hsh (wr_isSome_of_wr64 hw64 hj u))

/-- A read succeeds when `find` hits a region. -/
theorem rd_isSome_of_find {m : Mem} {a : Nat}
    (h : ∃ i r, Mem.find m a 0 = some (i, r)) :
    (m.rd a).isSome := by
  obtain ⟨i, r, hf⟩ := h
  obtain ⟨-, hle, hlt⟩ := find_some_mem hf
  have hidx : a - r.base < r.data.length := by
    unfold X86Env.Region.hi at hlt
    omega
  have hv : ∃ v, r.data[a - r.base]? = some v :=
    ⟨r.data[a - r.base]'hidx, List.getElem?_eq_getElem hidx⟩
  obtain ⟨v, hv⟩ := hv
  refine Option.isSome_iff_exists.mpr ⟨v, ?_⟩
  have he : m.rd a =
      (do let x ← Mem.find m a 0; x.2.data[a - x.2.base]?) := rfl
  rw [he, hf]
  exact hv

/-- An `rdN` from per-byte readability. -/
theorem rdN_isSome_of_forall_rd : ∀ (n : Nat) {m : Mem} {a : Nat},
    (∀ (j : Nat), j < n → (m.rd (a + j)).isSome) →
    (m.rdN a n).isSome := by
  intro n
  induction n with
  | zero => intro m a _; exact Option.isSome_iff_exists.mpr ⟨[], rfl⟩
  | succ n ih =>
    intro m a h
    have h0 := h 0 (by omega)
    obtain ⟨b, hb⟩ := Option.isSome_iff_exists.mp h0
    have hr := ih (a := a + 1) (fun j hj => by
      have hh := h (j + 1) (by omega)
      rwa [show a + 1 + j = a + (j + 1) by omega])
    obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hr
    exact Option.isSome_iff_exists.mpr
      ⟨b :: bs, rdN_cons_some hb hbs⟩

/-- An `rd64` from per-byte readability. -/
theorem rd64_some_of_forall_rd {m : Mem} {a : Nat}
    (h : ∀ (j : Nat), j < 8 → (m.rd (a + j)).isSome) :
    ∃ v, rd64 m a = some v := by
  have h8 := rdN_isSome_of_forall_rd 8 h
  obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp h8
  exact ⟨leVal bs, by rw [rd64, hbs]; rfl⟩

/-- `StackRoom` for a three-region memory whose second region
is the writable stack, disjoint from the first. -/
theorem stackRoom_of_c2 {s : X86Int.St} {c1 c2 buf : Region}
    {sp k : Nat}
    (hmem : s.mem = [c1, c2, buf]) (hr4 : s.reg 4 = some sp)
    (hrw : c2.rw = true) (hdisj : c1.hi ≤ c2.base)
    (hlo : c2.base + 8 * k ≤ sp) (hhi : sp ≤ c2.hi)
    (hspW : sp < W) :
    StackRoom s k := by
  have hfind : ∀ a : Nat, c2.base ≤ a → a < c2.hi →
      Mem.find [c1, c2, buf] a 0 = some (1, c2) := by
    intro a h1 h2
    have hg : Mem.find [c1, c2, buf] a 0 =
        (if c1.base ≤ a ∧ a < c1.hi then some (0, c1)
          else if c2.base ≤ a ∧ a < c2.hi then some (1, c2)
            else Mem.find [buf] a 2) := rfl
    rw [hg, if_neg (by omega), if_pos ⟨h1, h2⟩]
  intro sp' h4' i hi v
  have hsp' : sp' = sp := by
    rw [hr4] at h4'
    exact (Option.some.inj h4').symm
  rw [hsp']
  have hmod : (sp + W - 8 * (i + 1)) % W = sp - 8 * (i + 1) := by
    have hh : sp + W - 8 * (i + 1) = (sp - 8 * (i + 1)) + W := by
      omega
    rw [hh, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  rw [hmod]
  have hslot : c2.base ≤ sp - 8 * (i + 1) := by omega
  refine wr64_isSome_of_forall_wr (fun j hj u => ?_)
  rw [hmem]
  exact wr_isSome_iff.mpr ⟨1, c2,
    hfind _ (by omega) (by omega), hrw⟩

set_option maxRecDepth 8192 in
/-- Entry to `loaded` under the witness: the 21-step prefix, the
initial `mmap` (oracle-legal at its site) and the whole read
loop, composed.  The oracle at the end is the remaining
replicated placements over the table/scratch tail. -/
theorem loader_init_to_loaded {f path argv0 stdin : List Nat}
    {top : Nat} (hinit : Stage1InitOk f path argv0 top)
    (hroom : Room f path argv0 top) :
    ∃ A k s', A % page = 0 ∧ 0 < A ∧ A + stage1Mf f ≤ 2 ^ 47 ∧
      (stage1Init f path argv0 [] [] top).mem.free A (stage1Mf f) =
        true ∧
      run stage1Im k
        (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
          top)) = Out.ok s' ∧
      s'.rip = base + 0x134 ∧
      s'.reg 3 = some A ∧ (∃ cap', s'.reg 12 = some cap') ∧
      s'.reg 13 = some f.length ∧ s'.reg 14 = some 3 ∧
      s'.reg 4 =
        some ((X86Env.initStack top [argv0, path]).2 + 24) ∧
      (∃ v, s'.reg 0 = some v) ∧ (∃ v, s'.reg 10 = some v) ∧
      (∃ v, s'.reg 8 = some v) ∧ (∃ v, s'.reg 9 = some v) ∧
      s'.k.oracle =
        [.at (A + arenaFileBytes f),
         .at (A + (arenaFileBytes f + arenaTableBytes f))] ∧
      s'.k.fds = [some ⟨f, f.length⟩] ∧
      (∃ c1 c2 buf', s'.mem = [c1, c2, buf'] ∧
        c1.base = base ∧ c1.hi = base + stage1Code.length ∧
        c2.base = (X86Env.initStack top [argv0, path]).1.base ∧
        c2.hi = (X86Env.initStack top [argv0, path]).1.hi ∧
        buf'.base = A ∧ (∃ cap', buf'.hi = A + cap' ∧
          cap' ≤ arenaFileBytes f) ∧
        buf'.rw = true ∧ c2.rw = true) ∧
      s'.mem.rdN A f.length = some f ∧
      s'.regs.length = 16 ∧
      (∃ t, run stage1Im 21
        (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
          top)) = Out.ok t ∧ OracleLegalAt t) ∧
      StackRoom s' 10 := by
  obtain ⟨hfB, -, -, hpath0, -, hpathlen, -, htop47, hleInit,
    hstkhi⟩ := hinit
  obtain ⟨A, hAmod, hApos, hAbound, hfree⟩ := hroom
  have hdisjInit : base + stage1Code.length ≤
      (X86Env.initStack top [argv0, path]).1.base := by
    have h := hleInit
    rw [phVaddr_eq, phMemsz_eq] at h
    exact h
  have hk0 : (stage1Init f path argv0 stdin (stage1Oracle f A)
        top).k =
      ⟨[(path, f)], [], stdin, [], stage1Oracle f A⟩ := by
    simp [stage1Init]
  obtain ⟨s1, hrun1, hs1rip, hs1r0, hs1r7, hs1r6, hs1r2, hs1r10, hs1r8,
    hs1r9, hs1r14, hs1r4, hs1k, hshape, hs1len⟩ :=
    loader_entry_to_mmap
      (stk := (X86Env.initStack top [argv0, path]).1)
      (sp0 := (X86Env.initStack top [argv0, path]).2)
      (stdinL := stdin) (oracle := stage1Oracle f A)
      rfl rfl (stage1Init_rip _ _ _ _ _ _)
      (stage1Init_regs_length _ _ _ _ _ _)
      (stage1Init_reg4 _ _ _ _ _ _)
      (fun i hi4 hi16 => stage1Init_reg_ne4 _ _ _ _ _ _ hi16 hi4)
      (stage1Init_mem _ _ _ _ _ _) hk0 hdisjInit hstkhi htop47 hpath0
      hpathlen
  have hs1oracle : s1.k.oracle = stage1Oracle f A := by rw [hs1k]
  have hhead : s1.k.oracle.head? = some (.at A) := by
    rw [hs1oracle]
    simp [stage1Oracle, List.replicate_succ]
  have hfree1 : (stage1Init f path argv0 stdin (stage1Oracle f A)
        top).mem.free A (stage1Mf f) = true := by
    have hws : wshape (stage1Init f path argv0 stdin
        (stage1Oracle f A) top).mem =
        wshape (stage1Init f path argv0 [] [] top).mem := by
      rw [stage1Init_mem, stage1Init_mem]
    rw [free_of_wshape hws]
    exact hfree
  obtain ⟨hlegal, s2, hrun2, hs2rip, hs2rest⟩ :=
    loader_mmap1_to_readloop (s' := s1) hs1rip hs1len hs1r0
      hs1r7 hs1r6 hs1r2 hs1r10 hs1r8 hs1r9 hs1r14 hs1r4
      (run_reg_some_ne11 21 (by decide)
        (stage1Init_reg_ne4 _ _ _ _ _ _ (by decide) (by decide)) hrun1)
      hshape hhead hAmod hApos hAbound hfree1
  have hs2r3 := hs2rest.1
  have hs2r12 := hs2rest.2.1
  have hs2r13 := hs2rest.2.2.1
  have hs2r14 := hs2rest.2.2.2.1
  have hs2r4 := hs2rest.2.2.2.2.1
  have hs2r0 := hs2rest.2.2.2.2.2.1
  have hs2r10 := hs2rest.2.2.2.2.2.2.1
  have hs2r8 := hs2rest.2.2.2.2.2.2.2.1
  have hs2r9 := hs2rest.2.2.2.2.2.2.2.2.1
  have hs2mem := hs2rest.2.2.2.2.2.2.2.2.2.1
  have hs2k := hs2rest.2.2.2.2.2.2.2.2.2.2.1
  have hs2len := hs2rest.2.2.2.2.2.2.2.2.2.2.2
  have hshape2 : wshape s1.mem =
      [(base, base + stage1Code.length, false),
        ((X86Env.initStack top [argv0, path]).1.base,
          (X86Env.initStack top [argv0, path]).1.hi,
          (X86Env.initStack top [argv0, path]).1.rw)] := by
    rw [hshape]
    simp [stage1Init_mem, wshape, Region.hi]
  have hlen2 : s1.mem.length = 2 := by
    have h := congrArg List.length hshape2
    simp only [wshape, List.length_map, List.length_cons,
      List.length_nil] at h
    exact h
  obtain ⟨c1, c2, hs1mem⟩ : ∃ c1 c2, s1.mem = [c1, c2] := by
    cases hm : s1.mem with
    | nil => rw [hm] at hlen2; simp at hlen2
    | cons c1 rest =>
      cases rest with
      | nil => rw [hm] at hlen2; simp at hlen2
      | cons c2 rest2 =>
        cases rest2 with
        | nil => exact ⟨c1, c2, rfl⟩
        | cons c3 rest3 => rw [hm] at hlen2; simp at hlen2
  have hshape3 : ((c1.base, c1.hi, c1.rw) =
      (base, base + stage1Code.length, false)) ∧
      ((c2.base, c2.hi, c2.rw) =
        ((X86Env.initStack top [argv0, path]).1.base,
          (X86Env.initStack top [argv0, path]).1.hi,
          (X86Env.initStack top [argv0, path]).1.rw)) := by
    rw [hs1mem] at hshape2
    simp only [wshape, List.map_cons, List.map_nil,
      List.cons.injEq] at hshape2
    exact ⟨hshape2.1, hshape2.2.1⟩
  have hc1b : c1.base = base := by
    have h := hshape3.1
    simp only [Prod.mk.injEq] at h
    exact h.1
  have hc1hi : c1.hi = base + stage1Code.length := by
    have h := hshape3.1
    simp only [Prod.mk.injEq] at h
    exact h.2.1
  have hc2b : c2.base =
      (X86Env.initStack top [argv0, path]).1.base := by
    have h := hshape3.2
    simp only [Prod.mk.injEq] at h
    exact h.1
  have hc2hi : c2.hi =
      (X86Env.initStack top [argv0, path]).1.hi := by
    have h := hshape3.2
    simp only [Prod.mk.injEq] at h
    exact h.2.1
  have hs2mem' : s2.mem =
      [c1, c2, ⟨A, List.replicate 65536 0, true⟩] := by
    rw [hs2mem, hs1mem]
    rfl
  have hfreeC : ∀ r ∈ ([⟨base, stage1Code, false⟩,
        (X86Env.initStack top [argv0, path]).1] : Mem),
      r.hi ≤ A ∨ A + stage1Mf f ≤ r.base := by
    intro r hr
    have h := X86Env.free_spec hfree r (by
      have h2 : (stage1Init f path argv0 [] [] top).mem =
          [⟨base, stage1Code, false⟩,
            (X86Env.initStack top [argv0, path]).1] :=
        stage1Init_mem _ _ _ _ _ _
      rwa [h2])
    rcases h with h | h
    · exact Or.inr h
    · exact Or.inl h
  have hMfge : arenaFileBytes f ≤ stage1Mf f := by
    unfold stage1Mf
    omega
  have hdisj1 : c1.hi ≤ A ∨ A + arenaFileBytes f ≤ c1.base := by
    have h := hfreeC ⟨base, stage1Code, false⟩ List.mem_cons_self
    have hhi : (⟨base, stage1Code, false⟩ : Region).hi =
        base + stage1Code.length := by
      simp only [Region.hi]
    have hb : (⟨base, stage1Code, false⟩ : Region).base = base := rfl
    rw [hhi, hb] at h
    rcases h with h | h
    · exact Or.inl (by rw [hc1hi]; exact h)
    · exact Or.inr (by rw [hc1b]; omega)
  have hdisj2 : c2.hi ≤ A ∨ A + arenaFileBytes f ≤ c2.base := by
    have h := hfreeC (X86Env.initStack top [argv0, path]).1
      (List.mem_cons_of_mem _ List.mem_cons_self)
    rcases h with h | h
    · exact Or.inl (by rw [hc2hi]; exact h)
    · exact Or.inr (by rw [hc2b]; omega)
  have hnext2 : (Kern.next s1.k).2.oracle =
      List.replicate (stage1Growths f.length) (.at A) ++
        [.at (A + arenaFileBytes f),
         .at (A + (arenaFileBytes f + arenaTableBytes f))] := by
    have ho : s1.k.oracle = .at A ::
        (List.replicate (stage1Growths f.length) (.at A) ++
          [.at (A + arenaFileBytes f),
           .at (A + (arenaFileBytes f + arenaTableBytes f))]) := by
      rw [hs1oracle, stage1Oracle, List.replicate_succ,
        List.cons_append]
    cases ho2 : s1.k.oracle with
    | nil => simp [ho2] at ho
    | cons c rest =>
      have hnx : Kern.next s1.k =
          (some c, { s1.k with oracle := rest }) := by
        simp [Kern.next, ho2]
      rw [hnx]
      rw [ho2] at ho
      injection ho with h1 h2
  have hs2oracle : s2.k.oracle =
      List.replicate (stage1Growths f.length) (.at A) ++
        [.at (A + arenaFileBytes f),
         .at (A + (arenaFileBytes f + arenaTableBytes f))] := by
    rw [hs2k]
    exact hnext2
  have hs2fds : s2.k.fds = [some ⟨f, 0⟩] := by
    have hnx : (Kern.next s1.k).2.fds = s1.k.fds := by
      cases ho2 : s1.k.oracle with
      | nil => simp [Kern.next, ho2]
      | cons c rest => simp [Kern.next, ho2]
    rw [hs2k, hnx, hs1k]
  have hbuf0hi : (⟨A, List.replicate 65536 0, true⟩ : Region).hi =
      A + 65536 := by
    show A + (List.replicate 65536 0).length = A + 65536
    rw [List.length_replicate]
  have hA63 : A < 2 ^ 63 := by omega
  have hboundL : A + 65536 * 2 ^ (stage1Growths f.length) ≤ 2 ^ 47 := by
    have h1 := growths_cap_le_arena f
    omega
  obtain ⟨k, s3, hrun3, hs3rip, hs3r3, ⟨cap', hs3r12⟩, hs3r13, hs3r14,
    hs3r4, hs3r0, hs3r10, hs3r8, hs3r9, hs3oracle, hs3fds,
    ⟨buf3, hs3mem, hbuf3base, ⟨cap'', hbuf3hi, hbuf3bound⟩, hbuf3rw⟩,
    hs3content,
    hs3len⟩ :=
    loader_read_loop (f.length - 0) (by rw [Nat.sub_zero]) hs2rip
      hs2len hs2r3 hs2r12 hs2r13 hs2r14 hs2r4 hs2r0 hs2r10 hs2r8 hs2r9
      hs2mem' rfl rfl hbuf0hi hdisj1 hdisj2 (growths_cap_le_arena f)
      (by simp [Mem.rdN]) hs2fds hs2oracle ⟨A + arenaFileBytes f, rfl⟩
      hfB (Nat.zero_le _) (by decide) hAmod hApos hA63 (by decide)
      (by decide) hboundL (growths_cap_gt f.length)
      (growths_min f.length)
  have hrun : run stage1Im (21 + 6 + k)
      (Out.ok (stage1Init f path argv0 stdin (stage1Oracle f A)
        top)) = Out.ok s3 := by
    rw [show 21 + 6 + k = 21 + (6 + k) by omega, run_add stage1Im 21 _,
      hrun1, run_add stage1Im 6 k, hrun2, hrun3]
  exact ⟨A, 21 + 6 + k, s3, hAmod, hApos, hAbound, hfree, hrun, hs3rip,
    hs3r3, ⟨cap', hs3r12⟩, hs3r13, hs3r14, hs3r4, hs3r0, hs3r10, hs3r8,
    hs3r9, hs3oracle, hs3fds,
    ⟨c1, c2, buf3, hs3mem, hc1b, hc1hi, hc2b, hc2hi, hbuf3base,
      ⟨cap'', hbuf3hi, hbuf3bound⟩, hbuf3rw, by
        have h := hshape3.2
        simp only [Prod.mk.injEq] at h
        rw [h.2.2]
        exact rfl⟩, hs3content, hs3len,
    ⟨s1, hrun1, hlegal⟩, by
      have hc2rw : c2.rw = true := by
        have h := hshape3.2
        simp only [Prod.mk.injEq] at h
        rw [h.2.2]
        exact rfl
      have hsp0 : (X86Env.initStack top [argv0, path]).2 =
          c2.base + 4096 := by
        have h1 := initStack_sp top argv0 path
        rw [← hc2b] at h1
        have hsr : X86Env.stackRoom = 4096 := rfl
        rw [hsr] at h1
        exact h1
      have hhi2 : (X86Env.initStack top [argv0, path]).2 + 24 ≤
          c2.hi := by
        have hlen := initStack_data_length top argv0 path
        have hp : 4096 ≤ X86Env.pages
            (56 + ((argv0.length + 1) + (path.length + 1)) + 1) := by
          unfold X86Env.pages X86Env.page
          have h1 : 1 ≤ (56 + ((argv0.length + 1) +
              (path.length + 1)) + 1 + 4095) / 4096 :=
            Nat.div_pos (by omega) (by decide)
          omega
        have hh : c2.hi = c2.base +
            (X86Env.initStack top [argv0, path]).1.data.length := by
          have e : (X86Env.initStack top [argv0, path]).1.hi =
              (X86Env.initStack top [argv0, path]).1.base +
                (X86Env.initStack top [argv0, path]).1.data.length :=
            rfl
          rw [hc2hi, e, ← hc2b]
        have hsr : X86Env.stackRoom = 4096 := rfl
        omega
      refine stackRoom_of_c2 hs3mem hs3r4 hc2rw ?_ ?_ hhi2 ?_
      · rw [hc1hi, hc2b]; exact hdisjInit
      · omega
      · have hW : W = 2 ^ 64 := rfl
        have ht : c2.hi = top := by rw [hc2hi]; exact hstkhi
        omega⟩

/-- Layout of the parser region `base + 0x134` to `aCheckStart`
(`base + 0x452`): every instruction of the header checks, the two
allocation sites and the word-table decode, discharged against
the real image by `decide +kernel`. -/
structure ParserCode (im : Nat → Option (Ins × Nat)) : Prop where
  p134 : im (base + 0x134) = some (.mov 15 3, 3)
  p137 : im (base + 0x137) = some (.alu (.add) 15 13, 3)
  p13a : im (base + 0x13a) = some (.mov 6 3, 3)
  p13d : im (base + 0x13d) = some (.cmpI 13 16, 4)
  p141 : im (base + 0x141) = some (.jcc 3 (base + 0x148), 2)
  p143 : im (base + 0x143) = some (.jmp (base + 0x124), 5)
  p148 : im (base + 0x148) = some (.ld 0 6 0, 3)
  p14b : im (base + 0x14b) = some (.mov32 0 0, 2)
  p14d : im (base + 0x14d) = some (.cmpI 0 827478099, 7)
  p154 : im (base + 0x154) = some (.jcc 4 (base + 0x15b), 2)
  p156 : im (base + 0x156) = some (.jmp (base + 0x124), 5)
  p15b : im (base + 0x15b) = some (.mov 2 6, 3)
  p15e : im (base + 0x15e) = some (.aluI (.add) 2 4, 4)
  p162 : im (base + 0x162) = some (.ld 0 2 0, 3)
  p165 : im (base + 0x165) = some (.mov32 12 0, 3)
  p168 : im (base + 0x168) = some (.mov 2 6, 3)
  p16b : im (base + 0x16b) = some (.aluI (.add) 2 8, 4)
  p16f : im (base + 0x16f) = some (.ld 0 2 0, 3)
  p172 : im (base + 0x172) = some (.mov32 10 0, 3)
  p175 : im (base + 0x175) = some (.mov 2 6, 3)
  p178 : im (base + 0x178) = some (.aluI (.add) 2 12, 4)
  p17c : im (base + 0x17c) = some (.ld 0 2 0, 3)
  p17f : im (base + 0x17f) = some (.mov32 1 0, 2)
  p181 : im (base + 0x181) = some (.aluI (.add) 6 16, 4)
  p185 : im (base + 0x185) = some (.test 10 10, 3)
  p188 : im (base + 0x188) = some (.jcc 5 (base + 0x18f), 2)
  p18a : im (base + 0x18a) = some (.jmp (base + 0x124), 5)
  p18f : im (base + 0x18f) = some (.test 1 1, 3)
  p192 : im (base + 0x192) = some (.jcc 4 (base + 0x1c7), 2)
  p194 : im (base + 0x194) = some (.mov 2 15, 3)
  p197 : im (base + 0x197) = some (.alu (.sub) 2 6, 3)
  p19a : im (base + 0x19a) = some (.cmpI 2 4, 4)
  p19e : im (base + 0x19e) = some (.jcc 3 (base + 0x1a5), 2)
  p1a0 : im (base + 0x1a0) = some (.jmp (base + 0x124), 5)
  p1a5 : im (base + 0x1a5) = some (.ld32 0 6 0, 2)
  p1a7 : im (base + 0x1a7) = some (.mov 0 0, 3)
  p1aa : im (base + 0x1aa) = some (.aluI (.add) 6 4, 4)
  p1ae : im (base + 0x1ae) = some (.mov 2 15, 3)
  p1b1 : im (base + 0x1b1) = some (.alu (.sub) 2 6, 3)
  p1b4 : im (base + 0x1b4) = some (.alu (.sub) 2 0, 3)
  p1b7 : im (base + 0x1b7) = some (.jcc 3 (base + 0x1be), 2)
  p1b9 : im (base + 0x1b9) = some (.jmp (base + 0x124), 5)
  p1be : im (base + 0x1be) = some (.alu (.add) 6 0, 3)
  p1c1 : im (base + 0x1c1) = some (.aluI (.sub) 1 1, 4)
  p1c5 : im (base + 0x1c5) = some (.jmp (base + 0x18f), 2)
  p1c7 : im (base + 0x1c7) = some (.mov 0 10, 3)
  p1ca : im (base + 0x1ca) = some (.alu (.add) 0 0, 3)
  p1cd : im (base + 0x1cd) = some (.alu (.add) 0 0, 3)
  p1d0 : im (base + 0x1d0) = some (.mov 2 0, 3)
  p1d3 : im (base + 0x1d3) = some (.alu (.add) 0 0, 3)
  p1d6 : im (base + 0x1d6) = some (.alu (.add) 0 0, 3)
  p1d9 : im (base + 0x1d9) = some (.alu (.add) 0 2, 3)
  p1dc : im (base + 0x1dc) = some (.mov 2 15, 3)
  p1df : im (base + 0x1df) = some (.alu (.sub) 2 6, 3)
  p1e2 : im (base + 0x1e2) = some (.cmp 0 2, 3)
  p1e5 : im (base + 0x1e5) = some (.jcc 6 (base + 0x1ec), 2)
  p1e7 : im (base + 0x1e7) = some (.jmp (base + 0x124), 5)
  p1ec : im (base + 0x1ec) = some (.push 6, 1)
  p1ed : im (base + 0x1ed) = some (.mov32 2 10, 3)
  p1f0 : im (base + 0x1f0) = some (.alu (.add) 2 2, 3)
  p1f3 : im (base + 0x1f3) = some (.alu (.add) 2 2, 3)
  p1f6 : im (base + 0x1f6) = some (.alu (.add) 2 2, 3)
  p1f9 : im (base + 0x1f9) = some (.alu (.add) 2 2, 3)
  p1fc : im (base + 0x1fc) = some (.alu (.add) 2 2, 3)
  p1ff : im (base + 0x1ff) = some (.alu (.add) 2 2, 3)
  p202 : im (base + 0x202) = some (.movI 0 9, 5)
  p207 : im (base + 0x207) = some (.alu (.xor) 7 7, 2)
  p209 : im (base + 0x209) = some (.mov 6 2, 3)
  p20c : im (base + 0x20c) = some (.movI 2 3, 5)
  p211 : im (base + 0x211) = some (.movI 10 34, 6)
  p217 : im (base + 0x217) = some (.pushI 18446744073709551615, 2)
  p219 : im (base + 0x219) = some (.pop 8, 2)
  p21b : im (base + 0x21b) = some (.alu (.xor) 9 9, 3)
  p21e : im (base + 0x21e) = some (.syscall, 2)
  p220 : im (base + 0x220) = some (.test 0 0, 3)
  p223 : im (base + 0x223) = some (.jcc 9 (base + 0x22a), 2)
  p225 : im (base + 0x225) = some (.jmp (base + 0x12c), 5)
  p22a : im (base + 0x22a) = some (.mov 14 0, 3)
  p22d : im (base + 0x22d) = some (.movI 0 9, 5)
  p232 : im (base + 0x232) = some (.alu (.xor) 7 7, 2)
  p234 : im (base + 0x234) = some (.mov 6 13, 3)
  p237 : im (base + 0x237) = some (.movI 2 3, 5)
  p23c : im (base + 0x23c) = some (.movI 10 34, 6)
  p242 : im (base + 0x242) = some (.pushI 18446744073709551615, 2)
  p244 : im (base + 0x244) = some (.pop 8, 2)
  p246 : im (base + 0x246) = some (.alu (.xor) 9 9, 3)
  p249 : im (base + 0x249) = some (.syscall, 2)
  p24b : im (base + 0x24b) = some (.test 0 0, 3)
  p24e : im (base + 0x24e) = some (.jcc 9 (base + 0x255), 2)
  p250 : im (base + 0x250) = some (.jmp (base + 0x12c), 5)
  p255 : im (base + 0x255) = some (.mov 11 0, 3)
  p258 : im (base + 0x258) = some (.pop 6, 1)
  p259 : im (base + 0x259) = some (.mov 7 14, 3)
  p25c : im (base + 0x25c) = some (.ld 13 3 8, 4)
  p260 : im (base + 0x260) = some (.mov32 13 13, 3)
  p263 : im (base + 0x263) = some (.pop 0, 1)
  p264 : im (base + 0x264) = some (.pop 0, 1)
  p265 : im (base + 0x265) = some (.pop 0, 1)
  p266 : im (base + 0x266) = some (.pop 0, 1)
  p267 : im (base + 0x267) = some (.pop 0, 1)
  p268 : im (base + 0x268) = some (.pop 0, 1)
  p269 : im (base + 0x269) = some (.jmp (base + 0x2c9), 5)
  p26e : im (base + 0x26e) = some (.ld32 0 2 0, 2)
  p270 : im (base + 0x270) = some (.ret, 1)
  p271 : im (base + 0x271) = some (.mov 2 15, 3)
  p274 : im (base + 0x274) = some (.alu (.sub) 2 6, 3)
  p277 : im (base + 0x277) = some (.jcc 3 (base + 0x27e), 2)
  p279 : im (base + 0x279) = some (.jmp (base + 0x124), 5)
  p27e : im (base + 0x27e) = some (.cmpI 2 4, 4)
  p282 : im (base + 0x282) = some (.jcc 3 (base + 0x289), 2)
  p284 : im (base + 0x284) = some (.jmp (base + 0x124), 5)
  p289 : im (base + 0x289) = some (.mov 2 6, 3)
  p28c : im (base + 0x28c) = some (.call (base + 0x26e), 5)
  p291 : im (base + 0x291) = some (.aluI (.add) 6 4, 4)
  p295 : im (base + 0x295) = some (.ret, 1)
  p296 : im (base + 0x296) = some (.push 1, 1)
  p297 : im (base + 0x297) = some (.ld 0 7 48, 4)
  p29b : im (base + 0x29b) = some (.ld 1 7 32, 4)
  p29f : im (base + 0x29f) = some (.test 1 1, 3)
  p2a2 : im (base + 0x2a2) = some (.jcc 4 (base + 0x2c0), 2)
  p2a4 : im (base + 0x2a4) = some (.mov 2 0, 3)
  p2a7 : im (base + 0x2a7) = some (.push 0, 1)
  p2a8 : im (base + 0x2a8) = some (.call (base + 0x26e), 5)
  p2ad : im (base + 0x2ad) = some (.pop 2, 1)
  p2ae : im (base + 0x2ae) = some (.cmp 0 12, 3)
  p2b1 : im (base + 0x2b1) = some (.jcc 4 (base + 0x2c4), 2)
  p2b3 : im (base + 0x2b3) = some (.mov 0 2, 3)
  p2b6 : im (base + 0x2b6) = some (.aluI (.add) 0 12, 4)
  p2ba : im (base + 0x2ba) = some (.aluI (.sub) 1 1, 4)
  p2be : im (base + 0x2be) = some (.jmp (base + 0x29f), 2)
  p2c0 : im (base + 0x2c0) = some (.alu (.xor) 0 0, 2)
  p2c2 : im (base + 0x2c2) = some (.pop 1, 1)
  p2c3 : im (base + 0x2c3) = some (.ret, 1)
  p2c4 : im (base + 0x2c4) = some (.mov 0 2, 3)
  p2c7 : im (base + 0x2c7) = some (.pop 1, 1)
  p2c8 : im (base + 0x2c8) = some (.ret, 1)
  p2c9 : im (base + 0x2c9) = some (.mov 1 13, 3)
  p2cc : im (base + 0x2cc) = some (.test 1 1, 3)
  p2cf : im (base + 0x2cf) = some (.jcc 5 (base + 0x2d6), 2)
  p2d1 : im (base + 0x2d1) = some (.jmp (base + 0x419), 5)
  p2d6 : im (base + 0x2d6) = some (.call (base + 0x271), 5)
  p2db : im (base + 0x2db) = some (.st 7 0 0, 3)
  p2de : im (base + 0x2de) = some (.call (base + 0x271), 5)
  p2e3 : im (base + 0x2e3) = some (.st 7 8 0, 4)
  p2e7 : im (base + 0x2e7) = some (.call (base + 0x271), 5)
  p2ec : im (base + 0x2ec) = some (.st 7 16 0, 4)
  p2f0 : im (base + 0x2f0) = some (.call (base + 0x271), 5)
  p2f5 : im (base + 0x2f5) = some (.st 7 24 0, 4)
  p2f9 : im (base + 0x2f9) = some (.call (base + 0x271), 5)
  p2fe : im (base + 0x2fe) = some (.st 7 32 0, 4)
  p302 : im (base + 0x302) = some (.st 7 40 6, 4)
  p306 : im (base + 0x306) = some (.push 1, 1)
  p307 : im (base + 0x307) = some (.ld 1 7 24, 4)
  p30b : im (base + 0x30b) = some (.test 1 1, 3)
  p30e : im (base + 0x30e) = some (.jcc 5 (base + 0x315), 2)
  p310 : im (base + 0x310) = some (.jmp (base + 0x3db), 5)
  p315 : im (base + 0x315) = some (.mov 2 15, 3)
  p318 : im (base + 0x318) = some (.alu (.sub) 2 6, 3)
  p31b : im (base + 0x31b) = some (.jcc 3 (base + 0x322), 2)
  p31d : im (base + 0x31d) = some (.jmp (base + 0x124), 5)
  p322 : im (base + 0x322) = some (.cmpI 2 1, 4)
  p326 : im (base + 0x326) = some (.jcc 3 (base + 0x32d), 2)
  p328 : im (base + 0x328) = some (.jmp (base + 0x124), 5)
  p32d : im (base + 0x32d) = some (.ld8 0 6 0, 3)
  p330 : im (base + 0x330) = some (.cmpI 0 57, 4)
  p334 : im (base + 0x334) = some (.jcc 3 (base + 0x33b), 2)
  p336 : im (base + 0x336) = some (.jmp (base + 0x3ad), 5)
  p33b : im (base + 0x33b) = some (.cmpI 0 128, 7)
  p342 : im (base + 0x342) = some (.jcc 5 (base + 0x349), 2)
  p344 : im (base + 0x344) = some (.jmp (base + 0x3a5), 5)
  p349 : im (base + 0x349) = some (.cmpI 0 131, 7)
  p350 : im (base + 0x350) = some (.jcc 5 (base + 0x357), 2)
  p352 : im (base + 0x352) = some (.jmp (base + 0x3ad), 5)
  p357 : im (base + 0x357) = some (.cmpI 0 132, 7)
  p35e : im (base + 0x35e) = some (.jcc 5 (base + 0x365), 2)
  p360 : im (base + 0x360) = some (.jmp (base + 0x3ad), 5)
  p365 : im (base + 0x365) = some (.cmpI 0 133, 7)
  p36c : im (base + 0x36c) = some (.jcc 5 (base + 0x373), 2)
  p36e : im (base + 0x36e) = some (.jmp (base + 0x3ad), 5)
  p373 : im (base + 0x373) = some (.cmpI 0 140, 7)
  p37a : im (base + 0x37a) = some (.jcc 5 (base + 0x381), 2)
  p37c : im (base + 0x37c) = some (.jmp (base + 0x3ad), 5)
  p381 : im (base + 0x381) = some (.cmpI 0 129, 7)
  p388 : im (base + 0x388) = some (.jcc 3 (base + 0x38f), 2)
  p38a : im (base + 0x38a) = some (.jmp (base + 0x124), 5)
  p38f : im (base + 0x38f) = some (.cmpI 0 139, 7)
  p396 : im (base + 0x396) = some (.jcc 6 (base + 0x39d), 2)
  p398 : im (base + 0x398) = some (.jmp (base + 0x124), 5)
  p39d : im (base + 0x39d) = some (.movI 10 5, 6)
  p3a3 : im (base + 0x3a3) = some (.jmp (base + 0x3b3), 2)
  p3a5 : im (base + 0x3a5) = some (.movI 10 9, 6)
  p3ab : im (base + 0x3ab) = some (.jmp (base + 0x3b3), 2)
  p3ad : im (base + 0x3ad) = some (.movI 10 1, 6)
  p3b3 : im (base + 0x3b3) = some (.mov 2 15, 3)
  p3b6 : im (base + 0x3b6) = some (.alu (.sub) 2 6, 3)
  p3b9 : im (base + 0x3b9) = some (.cmp 2 10, 3)
  p3bc : im (base + 0x3bc) = some (.jcc 3 (base + 0x3c3), 2)
  p3be : im (base + 0x3be) = some (.jmp (base + 0x124), 5)
  p3c3 : im (base + 0x3c3) = some (.mov 2 6, 3)
  p3c6 : im (base + 0x3c6) = some (.alu (.sub) 2 3, 3)
  p3c9 : im (base + 0x3c9) = some (.alu (.add) 2 11, 3)
  p3cc : im (base + 0x3cc) = some (.st8 2 0 10, 3)
  p3cf : im (base + 0x3cf) = some (.alu (.add) 6 10, 3)
  p3d2 : im (base + 0x3d2) = some (.aluI (.sub) 1 1, 4)
  p3d6 : im (base + 0x3d6) = some (.jmp (base + 0x30b), 5)
  p3db : im (base + 0x3db) = some (.pop 1, 1)
  p3dc : im (base + 0x3dc) = some (.st 7 48 6, 4)
  p3e0 : im (base + 0x3e0) = some (.ld 0 7 32, 4)
  p3e4 : im (base + 0x3e4) = some (.mov 2 0, 3)
  p3e7 : im (base + 0x3e7) = some (.alu (.add) 0 0, 3)
  p3ea : im (base + 0x3ea) = some (.alu (.add) 0 0, 3)
  p3ed : im (base + 0x3ed) = some (.alu (.add) 2 2, 3)
  p3f0 : im (base + 0x3f0) = some (.alu (.add) 2 2, 3)
  p3f3 : im (base + 0x3f3) = some (.alu (.add) 2 2, 3)
  p3f6 : im (base + 0x3f6) = some (.alu (.add) 0 2, 3)
  p3f9 : im (base + 0x3f9) = some (.mov 2 15, 3)
  p3fc : im (base + 0x3fc) = some (.alu (.sub) 2 6, 3)
  p3ff : im (base + 0x3ff) = some (.cmp 2 0, 3)
  p402 : im (base + 0x402) = some (.jcc 3 (base + 0x409), 2)
  p404 : im (base + 0x404) = some (.jmp (base + 0x124), 5)
  p409 : im (base + 0x409) = some (.alu (.add) 6 0, 3)
  p40c : im (base + 0x40c) = some (.aluI (.add) 7 64, 4)
  p410 : im (base + 0x410) = some (.aluI (.sub) 1 1, 4)
  p414 : im (base + 0x414) = some (.jmp (base + 0x2cc), 5)
  p419 : im (base + 0x419) = some (.cmp 6 15, 3)
  p41c : im (base + 0x41c) = some (.jcc 4 (base + 0x423), 2)
  p41e : im (base + 0x41e) = some (.jmp (base + 0x124), 5)
  p423 : im (base + 0x423) = some (.cmp 12 13, 3)
  p426 : im (base + 0x426) = some (.jcc 2 (base + 0x42d), 2)
  p428 : im (base + 0x428) = some (.jmp (base + 0x124), 5)
  p42d : im (base + 0x42d) = some (.mov 2 12, 3)
  p430 : im (base + 0x430) = some (.alu (.add) 2 2, 3)
  p433 : im (base + 0x433) = some (.alu (.add) 2 2, 3)
  p436 : im (base + 0x436) = some (.alu (.add) 2 2, 3)
  p439 : im (base + 0x439) = some (.alu (.add) 2 2, 3)
  p43c : im (base + 0x43c) = some (.alu (.add) 2 2, 3)
  p43f : im (base + 0x43f) = some (.alu (.add) 2 2, 3)
  p442 : im (base + 0x442) = some (.alu (.add) 2 14, 3)
  p445 : im (base + 0x445) = some (.ld 0 2 0, 3)
  p448 : im (base + 0x448) = some (.test 0 0, 3)
  p44b : im (base + 0x44b) = some (.jcc 4 (base + 0x452), 2)
  p44d : im (base + 0x44d) = some (.jmp (base + 0x124), 5)

set_option maxHeartbeats 0 in
theorem stage1_parserCode : ParserCode stage1Im := by
  apply ParserCode.mk <;> decide +kernel
/-- The `fail2` block has the same status-exit shape as `fail1`,
with status 2. -/
theorem stage1_exitCode_fail2 : ExitCode stage1Im aFail2 2 := by
  constructor <;> decide +kernel

/-- Any state that reaches `fail2` with the registers defined and
stack room exits 2. -/
theorem exit2_of_fail2 {sp : Nat} (hsp32 : 32 ≤ sp) (hspW : sp < W) :
    ∀ s, s.rip = aFail2 → s.reg 4 = some sp → RegsOk s →
      StackRoom s 1 →
      ∃ k, run stage1Im k (Out.ok s) = Out.exit 2 := by
  intro s hrip hr4 hregs hroom
  obtain ⟨k, hk⟩ := Status_exit_of_room stage1_exitCode_fail2
    (by unfold W; decide) (by omega) hspW s hrip hr4 hregs hroom
  exact ⟨k, by simpa using hk⟩

/-! ### Parser segment infrastructure -/

theorem rdN_length {m : Mem} : ∀ {a n : Nat} {bs : List Nat},
    m.rdN a n = some bs → bs.length = n := by
  intro a n
  induction n generalizing a with
  | zero =>
    intro bs h
    have h2 : m.rdN a 0 = some [] := rfl
    rw [h2] at h
    have hbs : bs = [] := (Option.some.inj h).symm
    rw [hbs]; rfl
  | succ n ih =>
    intro bs h
    cases bs with
    | nil =>
      exfalso
      have h1 : (do let b ← m.rd a; let r ← Mem.rdN m (a + 1) n; pure (b :: r)) = some ([] : List Nat) := h
      obtain ⟨b, hb, h2⟩ := bind_some h1
      obtain ⟨r, hr, h3⟩ := bind_some h2
      simp at h3
    | cons b bs =>
      obtain ⟨-, hr⟩ := rdN_cons h
      rw [List.length_cons, ih hr]

theorem rdN_slice {m : Mem} {a n o k : Nat} {bs : List Nat}
    (h : m.rdN a n = some bs) (hok : o + k ≤ n) :
    m.rdN (a + o) k = some ((bs.drop o).take k) := by
  have hlen : bs.length = n := rdN_length h
  have hsplit1 : bs = bs.take o ++ bs.drop o :=
    (List.take_append_drop o bs).symm
  have h1 : m.rdN a ((bs.take o).length + (bs.drop o).length) =
      some (bs.take o ++ bs.drop o) := by
    have hl : (bs.take o).length + (bs.drop o).length = n := by
      rw [← List.length_append, List.take_append_drop]; exact hlen
    rw [hl, ← hsplit1]; exact h
  obtain ⟨-, h2⟩ := rdN_append h1
  have htake : (bs.take o).length = o := by
    rw [List.length_take, hlen, Nat.min_eq_left (by omega)]
  rw [htake] at h2
  have hsplit2 : bs.drop o = (bs.drop o).take k ++ (bs.drop o).drop k :=
    (List.take_append_drop k (bs.drop o)).symm
  have h3 : m.rdN (a + o)
      (((bs.drop o).take k).length + ((bs.drop o).drop k).length) =
      some ((bs.drop o).take k ++ (bs.drop o).drop k) := by
    have hl : ((bs.drop o).take k).length + ((bs.drop o).drop k).length =
        (bs.drop o).length := by
      rw [← List.length_append, List.take_append_drop]
    rw [hl, ← hsplit2]; exact h2
  obtain ⟨h4, -⟩ := rdN_append h3
  have hklen : ((bs.drop o).take k).length = k := by
    rw [List.length_take, List.length_drop, hlen,
      Nat.min_eq_left (by omega)]
  rw [hklen] at h4
  exact h4

theorem leVal_take4_mod32 {xs : List Nat} (h8 : xs.length = 8) :
    leVal xs % 2 ^ 32 = leVal (xs.take 4) % 2 ^ 32 := by
  have hsplit : xs = xs.take 4 ++ xs.drop 4 :=
    (List.take_append_drop 4 xs).symm
  have hlen4 : (xs.take 4).length = 4 := by
    rw [List.length_take, h8]; decide
  have h256 : 256 ^ 4 = 2 ^ 32 := by decide
  have hdecomp := leVal_append (xs.take 4) (xs.drop 4)
  rw [← hsplit, hlen4, h256] at hdecomp
  rw [hdecomp, Nat.add_mul_mod_self_left]

theorem find_append_left : ∀ (m : Mem) (t : Mem) (a i : Nat)
    {j : Nat} {r : Region},
    Mem.find m a i = some (j, r) → Mem.find (m ++ t) a i = some (j, r) := by
  intro m
  induction m with
  | nil =>
    intro t a i j r h
    simp [Mem.find] at h
  | cons r0 rest ih =>
    intro t a i j r h
    have h1 : (if r0.base ≤ a ∧ a < r0.hi then some (i, r0)
        else Mem.find rest a (i + 1)) = some (j, r) := h
    by_cases hcond : r0.base ≤ a ∧ a < r0.hi
    · rw [if_pos hcond] at h1
      have hg : Mem.find ((r0 :: rest) ++ t) a i =
          (if r0.base ≤ a ∧ a < r0.hi then some (i, r0)
            else Mem.find (rest ++ t) a (i + 1)) := rfl
      rw [hg, if_pos hcond]; exact h1
    · rw [if_neg hcond] at h1
      have hg : Mem.find ((r0 :: rest) ++ t) a i =
          (if r0.base ≤ a ∧ a < r0.hi then some (i, r0)
            else Mem.find (rest ++ t) a (i + 1)) := rfl
      rw [hg, if_neg hcond]; exact ih t a (i + 1) h1

theorem rd_append_left {m : Mem} {t : Mem} {a b : Nat}
    (h : m.rd a = some b) : (m ++ t).rd a = some b := by
  have h1 : (do let x ← Mem.find m a 0; x.2.data[a - x.2.base]?) =
      some b := h
  obtain ⟨⟨j, r⟩, hf, hb⟩ := bind_some h1
  have hg : (m ++ t).rd a =
      (do let x ← Mem.find (m ++ t) a 0; x.2.data[a - x.2.base]?) := rfl
  rw [hg, find_append_left m t a 0 hf]
  exact hb

theorem rdN_append_left {m : Mem} {t : Mem} :
    ∀ {a n : Nat} {bs : List Nat},
    m.rdN a n = some bs → (m ++ t).rdN a n = some bs := by
  intro a n
  induction n generalizing a with
  | zero =>
    intro bs h
    have h2 : m.rdN a 0 = some [] := rfl
    rw [h2] at h
    have hbs : bs = [] := (Option.some.inj h).symm
    rw [hbs]; rfl
  | succ n ih =>
    intro bs h
    cases bs with
    | nil =>
      exfalso
      have h1 : (do let b ← m.rd a; let r ← Mem.rdN m (a + 1) n; pure (b :: r)) = some ([] : List Nat) := h
      obtain ⟨b, hb, h2⟩ := bind_some h1
      obtain ⟨r, hr, h3⟩ := bind_some h2
      simp at h3
    | cons b bs =>
      obtain ⟨hb, hr⟩ := rdN_cons h
      exact rdN_cons_some (rd_append_left hb) (ih hr)

theorem syscall_mmap_exact {k : Kern} {m : Mem} {a2 A : Nat}
    (ha2 : a2 ≠ 0) (hhead : k.oracle.head? = some (.at A))
    (hplace : X86Env.place m (some (.at A)) (X86Env.pages a2) =
      some (some A)) :
    X86Env.syscall k m 9 0 a2 PROT_RW MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret A (Kern.next k).2
        (m ++ [⟨A, List.replicate (X86Env.pages a2) 0, true⟩])) := by
  have hnext1 : (Kern.next k).1 = some (.at A) := kern_next_of_head hhead
  simp only [X86Env.syscall]
  rw [if_neg (by simp), if_neg ha2]
  cases hnext : Kern.next k with
  | mk c k1 =>
    rw [hnext] at hnext1
    simp only [] at hnext1
    subst hnext1
    simp [hplace]

/-- The `RDC` body at `aRDC`, success case, with the frame the
parser composition needs (the exported `RDCSpec` post is too weak
on preserved registers): only `rax`, `rdx`, `rsp` and `rsi`
change, and memory changes only in the one stack slot of the
inner `call` to `L32`.  The exit is the `ret` instruction at
`aRDC + 0x24`, with `rsp` still decremented by the caller's
`call`. -/
theorem rdc_body_ok {c e sp : Nat} {s : X86Int.St}
    (hrip : s.rip = aRDC) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : (s.mem.rdN c 4).isSome) (hroom : StackRoom s 1)
    (hr4 : s.reg 4 = some sp) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hframe : ∀ m, wr64 s.mem ((sp + W - 8) % W) (aRDC + 32) = some m →
      m.rdN c 4 = s.mem.rdN c 4) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aRDC + 0x24 ∧
      s'.reg 0 = (s.mem.rdN c 4).map leVal ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.mem.rdN c 4 = s.mem.rdN c 4 ∧ s'.k = s.k ∧
      (∃ m1, wr64 s.mem (sp - 8) (aRDC + 32) = some m1 ∧
        s'.mem = m1) := by
  have hc := stage1_parserCode
  have haRDC : aRDC = base + 0x271 := rfl
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- step 1: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip, haRDC]; exact hc.p271) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x274 := by
    rw [ht1]; show s.rip + 3 = base + 0x274; rw [hrip, haRDC]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- step 2: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p274) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht2rip : t2.rip = base + 0x277 := by
    rw [ht2]; show t1.rip + 3 = base + 0x277; rw [ht1rip]
  have ht2r2 : t2.reg 2 = some (e - c) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht1len]; decide), hsub]
  have ht2r6 : t2.reg 6 = some c := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht1r6
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have ht2fl : X86Int.cond t2.fl 3 = some true := by
    have hcf : t2.fl.cf = some false := by
      rw [ht2]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- step 3: jae 0x27e
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x27e } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p277) ht2fl
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x27e } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x27e := by rw [ht3]
  have ht3r2 : t3.reg 2 = some (e - c) := by
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).reg 2 = some (e - c)
    rw [reg_with_rip]; exact ht2r2
  have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).reg i = s.reg i
    rw [reg_with_rip]; exact ht2fr i hi
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).mem = s.mem
    rw [mem_with_rip]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2 with rip := base + 0x27e }).regs.length = 16
    rw [len_with_rip]; exact ht2len
  -- step 4: cmp rdx, 4
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p27e) ht3r2
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
    ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x282 := by
    rw [ht4]; show t3.rip + 4 = base + 0x282; rw [ht3rip]
  have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some true := by
    have hcf : t4.fl.cf = some false := by
      rw [ht4]
      show (aluF .sub (e - c) (4 % W)).2.cf = some false
      rw [h4W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht4fr : ∀ i, i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi
    rw [ht4]
    show t3.reg i = s.reg i
    exact ht3fr i hi
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show t3.mem = s.mem
    exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]; show t3.regs.length = 16; exact ht3len
  -- step 5: jae 0x289
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x289 } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p282) ht4fl
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x289 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x289 := by rw [ht5]
  have ht5r6 : t5.reg 6 = some c := by
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).reg 6 = some c
    rw [reg_with_rip, ht4fr 6 (by decide)]; exact hr6
  have ht5r4 : t5.reg 4 = some sp := by
    have h := ht4fr 4 (by decide)
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).reg 4 = some sp
    rw [reg_with_rip]
    rw [h]; exact hr4
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht4fr i hi
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).mem = s.mem
    rw [mem_with_rip]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4 with rip := base + 0x289 }).regs.length = 16
    rw [len_with_rip]; exact ht4len
  -- step 6: mov rdx, rsi
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 2 (some c) with rip := t5.rip + 3 } :=
    step_mov (by rw [ht5rip]; exact hc.p289) ht5r6
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 2 (some c) with rip := t5.rip + 3 } := ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x28c := by
    rw [ht6]; show t5.rip + 3 = base + 0x28c; rw [ht5rip]
  have ht6r2 : t6.reg 2 = some c := by
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (by rw [ht5len]; decide)]
  have ht6fr : ∀ i, i ≠ 2 → t6.reg i = s.reg i := by
    intro i hi
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht5fr i hi
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht5len
  have ht6r4 : t6.reg 4 = some sp := by
    have h := ht6fr 4 (by decide)
    rw [h]; exact hr4
  -- step 7: call L32
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W)
      (aRDC + 32) = some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (aRDC + 32))
  have hm1 : wr64 t6.mem ((sp + W - 8) % W) (aRDC + 32) = some m1 := by
    rw [ht6mem]; exact hm1raw
  have hm1' : wr64 t6.mem ((sp + W - 8) % W) (t6.rip + 5) = some m1 := by
    have h32 : aRDC + 32 = t6.rip + 5 := by rw [ht6rip, haRDC]
    rw [← h32]; exact hm1
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aL32 } := by
    have h := step_call (im := stage1Im) (s := t6) (m := m1)
      (by rw [ht6rip]; exact hc.p28c) ht6r4 hm1'
    simpa using h
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aL32 } :=
    ⟨_, rfl⟩
  have ht7rip : t7.rip = aL32 := by rw [ht7]
  have ht7r2 : t7.reg 2 = some c := by
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).reg 2 = some c
    rw [reg_set_other (by decide)]; exact ht6r2
  have ht7fr : ∀ i, i ≠ 2 → i ≠ 4 → t7.reg i = s.reg i := by
    intro i hi2 hi4
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]; exact ht6fr i hi2
  have ht7mem : t7.mem = m1 := by rw [ht7]
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact ht6len
  have hm1rd : m1.rdN c 4 = s.mem.rdN c 4 := by
    apply hframe
    rw [← ht6mem]; exact hm1
  -- L32 body: one ld32 step to its ret instruction
  obtain ⟨bs, hbs⟩ : ∃ bs, s.mem.rdN c 4 = some bs :=
    Option.isSome_iff_exists.mp hrd
  have hbs7 : t7.mem.rdN c 4 = some bs := by
    rw [ht7mem, hm1rd]; exact hbs
  have hrd32 : rd32 t7.mem (ea c 0) = some (leVal bs) := by
    unfold rd32 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hcW, hbs7]; rfl
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 } :=
    step_ld32 (by rw [ht7rip]; exact stage1_l32.d0) ht7r2 hrd32
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 } := ⟨_, rfl⟩
  have hrun8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have ht8rip : t8.rip = aL32 + 2 := by
    rw [ht8]; show t7.rip + 2 = aL32 + 2; rw [ht7rip]
  have ht8r0v : t8.reg 0 = (s.mem.rdN c 4).map leVal := by
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_with_rip, reg_set_self (by rw [ht7len]; decide), hbs]; rfl
  have ht8mem : t8.mem = t7.mem := by
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).mem =
      t7.mem
    rw [mem_with_rip, mem_set]
  have ht8fr : ∀ i, i ≠ 0 → t8.reg i = t7.reg i := by
    intro i hi
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).reg i =
      t7.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show (t7.set 0 (some (leVal bs))).regs.length = 16
    rw [len_set]; exact ht7len
  have ht8k : t8.k = s.k := by
    have h7k : t7.k = s.k := by
      rw [ht7]
      show (t6.set 4 (some ((sp + W - 8) % W))).k = s.k
      have h6k : t6.k = s.k := by
        rw [ht6]
        show ({ t5.set 2 (some c) with rip := t5.rip + 3 }).k = s.k
        have h5k : t5.k = s.k := by
          rw [ht5]
          show ({ t4 with rip := base + 0x289 }).k = s.k
          have h4k : t4.k = s.k := by
            rw [ht4]
            show t3.k = s.k
            rw [ht3]
            show ({ t2 with rip := base + 0x27e }).k = s.k
            exact ht2k
          exact h4k
        exact h5k
      exact h6k
    rw [ht8]
    show ({ t7.set 0 (some (leVal bs)) with rip := t7.rip + 2 }).k = s.k
    exact h7k
  -- step: ret from L32
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have hW : W = 2 ^ 64 := rfl
    have : sp + W - 8 = (sp - 8) + W := by omega
    rw [this, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht8r4 : t8.reg 4 = some (sp - 8) := by
    have h := ht8fr 4 (by decide)
    rw [h]
    rw [ht7]
    show (t6.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by rw [ht6len]; decide), hsp8mod]
  have hrd64slot : rd64 t8.mem (sp - 8) = some (aRDC + 32) := by
    rw [ht8mem, ht7mem]
    have h := hm1
    rw [ht6mem, hsp8mod] at h
    exact rd64_wr64 h
  have haL32 : aL32 = base + 0x26e := rfl
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 4 (some ((sp - 8 + 8) % W)) with rip := aRDC + 32 } :=
    step_ret (im := stage1Im) (by rw [ht8rip, haL32]; exact hc.p270)
      ht8r4 hrd64slot
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 4 (some ((sp - 8 + 8) % W)) with rip := aRDC + 32 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x291 := by
    rw [ht9]; show aRDC + 32 = base + 0x291; rw [haRDC]
  have hspret : (sp - 8 + 8) % W = sp := by
    have hW : W = 2 ^ 64 := rfl
    rw [show sp - 8 + 8 = sp by omega, Nat.mod_eq_of_lt hspW]
  have ht9r4 : t9.reg 4 = some sp := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg 4 = some sp
    rw [reg_set_self (by rw [ht8len]; decide), hspret]
  have ht9r6 : t9.reg 6 = some c := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]
    rw [ht8fr 6 (by decide), ht7fr 6 (by decide) (by decide)]
    exact hr6
  have ht9r0 : t9.reg 0 = (s.mem.rdN c 4).map leVal := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide)]
    exact ht8r0v
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → t9.reg i = s.reg i := by
    intro i hi0 hi2 hi4 _
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]
    rw [ht8fr i hi0]
    exact ht7fr i hi2 hi4
  have ht9mem : t9.mem = m1 := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).mem = m1
    rw [mem_set, ht8mem, ht7mem]
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 4 (some ((sp - 8 + 8) % W))).regs.length = 16
    rw [len_set]; exact ht8len
  -- step: add rsi, 4
  have hadd4 : (aluF .add c (4 % W)).1 = c + 4 := by
    rw [h4W, aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t9.rip + 4 } :=
    step_aluI_add (by rw [ht9rip]; exact hc.p291) ht9r6
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t9.rip + 4 } := ⟨_, rfl⟩
  -- compose the run
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have hrun : run stage1Im 10 (Out.ok s) = Out.ok t10 := by
    rw [show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r1,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r2,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r3,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r4,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r5,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r6,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r7,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, hrun8,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r9, r10]
  have hm1wr : wr64 s.mem (sp - 8) (aRDC + 32) = some m1 := by
    have h := hm1
    rwa [ht6mem, hsp8mod] at h
  refine ⟨10, t10, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht10]
    show t9.rip + 4 = aRDC + 0x24
    rw [ht9rip, haRDC]
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide)]; exact ht9r0
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg 6 = some (c + 4)
    rw [reg_set_self (by rw [ht9len]; decide), hadd4]
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg 4 = some sp
    rw [reg_set_other (by decide)]; exact ht9r4
  · intro i hi0 hi2 hi4 hi6
    rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi6]; exact ht9fr i hi0 hi2 hi4 hi6
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).regs.length = 16
    rw [len_set]; exact ht9len
  · intro x k hdis
    rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).mem.rdN x k =
      s.mem.rdN x k
    rw [mem_set, ht9mem]
    exact rdN_wr64_other hm1wr (by
      rcases hdis with h | h
      · exact Or.inr h
      · exact Or.inl (by omega))
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).mem.rdN c 4 =
      s.mem.rdN c 4
    rw [mem_set, ht9mem]
    exact hm1rd
  · rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).k = s.k
    have h9k : t9.k = s.k := by
      rw [ht9]
      show (t8.set 4 (some ((sp - 8 + 8) % W))).k = s.k
      exact ht8k
    exact h9k
  · refine ⟨m1, hm1wr, ?_⟩
    rw [ht10]
    show (t9.set 6 (some (aluF .add c (4 % W)).1)).mem = m1
    rw [mem_set]; exact ht9mem

/-- The `RDC` body at `aRDC`, refusal case: fewer than four
bytes between cursor and end sends the run to `aFail1`, touching
only `rdx`. -/
theorem rdc_body_fail {c e : Nat} {s : X86Int.St}
    (hrip : s.rip = aRDC) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : ¬ c + 4 ≤ e)
    (hcW : c < W) (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧
      (∀ i, i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have haRDC : aRDC = base + 0x271 := rfl
  have haFail1 : aFail1 = base + 0x124 := rfl
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- step 1: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip, haRDC]; exact hc.p271) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x274 := by
    rw [ht1]; show s.rip + 3 = base + 0x274; rw [hrip, haRDC]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- step 2: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p274) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x277 := by
    rw [ht2]; show t1.rip + 3 = base + 0x277; rw [ht1rip]
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  by_cases hec : e < c
  · -- first jae not taken: fall to the jmp at 0x279
    have ht2fl : X86Int.cond t2.fl 3 = some false := by
      have hcf : t2.fl.cf = some true := by
        rw [ht2]; exact aluF_sub_cf_true hec
      have hcc := cond_cc3 hcf
      simpa using hcc
    have e3 : X86Int.step stage1Im t2 =
        .ok { t2 with rip := base + 0x279 } := by
      have h := step_jcc (im := stage1Im) (s := t2)
        (by rw [ht2rip]; exact hc.p277) ht2fl
      simpa [ht2rip] using h
    obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
        { t2 with rip := base + 0x279 } := ⟨_, rfl⟩
    have ht3rip : t3.rip = base + 0x279 := by rw [ht3]
    have e4 : X86Int.step stage1Im t3 =
        .ok { t3 with rip := base + 0x124 } :=
      step_jmp (by rw [ht3rip]; exact hc.p279)
    obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
        { t3 with rip := base + 0x124 } := ⟨_, rfl⟩
    have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
      rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
    have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
      rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
    have hrun : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
      rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
        show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
    refine ⟨4, t4, hrun, ?_, ?_, ?_, ?_, ?_⟩
    · rw [ht4]; show base + 0x124 = aFail1; rw [haFail1]
    · intro i hi
      have h3i : t3.reg i = s.reg i := by
        rw [ht3]
        show ({ t2 with rip := base + 0x279 }).reg i = s.reg i
        rw [reg_with_rip]; exact ht2fr i hi
      rw [ht4]
      show ({ t3 with rip := base + 0x124 }).reg i = s.reg i
      rw [reg_with_rip]; exact h3i
    · rw [ht4]
      show ({ t3 with rip := base + 0x124 }).mem = s.mem
      rw [mem_with_rip]
      rw [ht3]
      show ({ t2 with rip := base + 0x279 }).mem = s.mem
      rw [mem_with_rip]; exact ht2mem
    · rw [ht4]
      show ({ t3 with rip := base + 0x124 }).regs.length = 16
      rw [len_with_rip]
      rw [ht3]
      show ({ t2 with rip := base + 0x279 }).regs.length = 16
      rw [len_with_rip]; exact ht2len
    · rw [ht4]
      show ({ t3 with rip := base + 0x124 }).k = s.k
      rw [ht3]
      show ({ t2 with rip := base + 0x279 }).k = s.k
      exact ht2k
  · -- c ≤ e < c + 4: second jae not taken: jmp at 0x284
    have hcle : c ≤ e := by omega
    have hsub : (aluF .sub e c).1 = e - c := by
      rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
        Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
    have ht2r2 : t2.reg 2 = some (e - c) := by
      rw [ht2]
      show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
      rw [reg_set_self (by rw [ht1len]; decide), hsub]
    have ht2fl : X86Int.cond t2.fl 3 = some true := by
      have hcf : t2.fl.cf = some false := by
        rw [ht2]; exact aluF_sub_cf (by omega)
      have hcc := cond_cc3 hcf
      simpa using hcc
    have e3 : X86Int.step stage1Im t2 =
        .ok { t2 with rip := base + 0x27e } := by
      have h := step_jcc (im := stage1Im) (s := t2)
        (by rw [ht2rip]; exact hc.p277) ht2fl
      simpa using h
    obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
        { t2 with rip := base + 0x27e } := ⟨_, rfl⟩
    have ht3rip : t3.rip = base + 0x27e := by rw [ht3]
    have ht3r2 : t3.reg 2 = some (e - c) := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).reg 2 = some (e - c)
      rw [reg_with_rip]; exact ht2r2
    have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
      intro i hi
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).reg i = s.reg i
      rw [reg_with_rip]; exact ht2fr i hi
    have ht3mem : t3.mem = s.mem := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).mem = s.mem
      rw [mem_with_rip]; exact ht2mem
    have ht3len : t3.regs.length = 16 := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).regs.length = 16
      rw [len_with_rip]; exact ht2len
    have ht3k : t3.k = s.k := by
      rw [ht3]
      show ({ t2 with rip := base + 0x27e }).k = s.k
      exact ht2k
    -- cmp rdx, 4
    have e4 : X86Int.step stage1Im t3 =
        .ok { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
      step_cmpI (by rw [ht3rip]; exact hc.p27e) ht3r2
    obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
        { t3 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t3.rip + 4 } :=
      ⟨_, rfl⟩
    have ht4rip : t4.rip = base + 0x282 := by
      rw [ht4]; show t3.rip + 4 = base + 0x282; rw [ht3rip]
    have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
    have ht4fl : X86Int.cond t4.fl 3 = some false := by
      have hcf : t4.fl.cf = some true := by
        rw [ht4]
        show (aluF .sub (e - c) (4 % W)).2.cf = some true
        rw [h4W]; exact aluF_sub_cf_true (by omega)
      have hcc := cond_cc3 hcf
      simpa using hcc
    have e5 : X86Int.step stage1Im t4 =
        .ok { t4 with rip := base + 0x284 } := by
      have h := step_jcc (im := stage1Im) (s := t4)
        (by rw [ht4rip]; exact hc.p282) ht4fl
      simpa [ht4rip] using h
    obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
        { t4 with rip := base + 0x284 } := ⟨_, rfl⟩
    have ht5rip : t5.rip = base + 0x284 := by rw [ht5]
    have e6 : X86Int.step stage1Im t5 =
        .ok { t5 with rip := base + 0x124 } :=
      step_jmp (by rw [ht5rip]; exact hc.p284)
    obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
        { t5 with rip := base + 0x124 } := ⟨_, rfl⟩
    have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
      rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
    have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
      rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
    have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
      rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
    have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
      rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
    have hrun : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
      rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1,
        show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2,
        show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3,
        show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
    refine ⟨6, t6, hrun, ?_, ?_, ?_, ?_, ?_⟩
    · rw [ht6]; show base + 0x124 = aFail1; rw [haFail1]
    · intro i hi
      have h4i : t4.reg i = s.reg i := by
        rw [ht4]
        show t3.reg i = s.reg i
        exact ht3fr i hi
      have h5i : t5.reg i = s.reg i := by
        rw [ht5]
        show ({ t4 with rip := base + 0x284 }).reg i = s.reg i
        rw [reg_with_rip]; exact h4i
      rw [ht6]
      show ({ t5 with rip := base + 0x124 }).reg i = s.reg i
      rw [reg_with_rip]; exact h5i
    · rw [ht6]
      show ({ t5 with rip := base + 0x124 }).mem = s.mem
      rw [mem_with_rip]
      rw [ht5]
      show ({ t4 with rip := base + 0x284 }).mem = s.mem
      rw [mem_with_rip]
      rw [ht4]
      show t3.mem = s.mem
      exact ht3mem
    · rw [ht6]
      show ({ t5 with rip := base + 0x124 }).regs.length = 16
      rw [len_with_rip]
      rw [ht5]
      show ({ t4 with rip := base + 0x284 }).regs.length = 16
      rw [len_with_rip]
      rw [ht4]
      show t3.regs.length = 16
      exact ht3len
    · rw [ht6]
      show ({ t5 with rip := base + 0x124 }).k = s.k
      rw [ht5]
      show ({ t4 with rip := base + 0x284 }).k = s.k
      rw [ht4]
      show t3.k = s.k
      exact ht3k

/-- A parser `call RDC` site, success case: from the call
instruction to the instruction after it, with `rax` the `u32` at
the cursor, the cursor advanced by four, `rsp` restored, and
memory changed only in the two stack slots `[sp-16, sp)`. -/
theorem call_rdc_ok {site c e sp : Nat} {s : X86Int.St}
    (hsite : stage1Im site = some (.call aRDC, 5)) (hsiteW : site + 5 < W)
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : (s.mem.rdN c 4).isSome) (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hdisj : c + 4 ≤ sp - 16 ∨ sp ≤ c) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 5 ∧
      s'.reg 0 = (s.mem.rdN c 4).map leVal ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧
      (∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.mem.rdN c 4 = s.mem.rdN c 4 ∧ s'.k = s.k ∧
      (∃ m1 m2, wr64 s.mem (sp - 8) (site + 5) = some m1 ∧
        wr64 m1 (sp - 8 - 8) (aRDC + 32) = some m2 ∧ s'.mem = m2) := by
  have hc := stage1_parserCode
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hsp16mod : (sp + W - 16) % W = sp - 16 := by
    have h : sp + W - 16 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have haddr2 : (sp - 8 + W - 8) % W = sp - 16 := by
    have h : sp - 8 + W - 8 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  -- the call step
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W)
      (site + 5) = some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (site + 5))
  have hm1 : wr64 s.mem ((sp + W - 8) % W) (site + 5) = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    rwa [hsp8mod] at hm1
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    step_call (im := stage1Im) (s := s) (m := m1)
      (by rw [hrip]; exact hsite) hr4 (by rw [hrip]; exact hm1)
  obtain ⟨s1, hs1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    ⟨_, rfl⟩
  have hs1rip : s1.rip = aRDC := by rw [hs1]
  have hs1r6 : s1.reg 6 = some c := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 15 = some e
    rw [reg_set_other (by decide)]; exact hr15
  have hs1r4 : s1.reg 4 = some (sp - 8) := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by rw [hlen16]; decide), hsp8mod]
  have hs1fr : ∀ i, i ≠ 4 → s1.reg i = s.reg i := by
    intro i hi
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have hs1mem : s1.mem = m1 := by rw [hs1]
  have hs1len : s1.regs.length = 16 := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have hs1k : s1.k = s.k := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).k = s.k
    rfl
  have hm1rd : m1.rdN c 4 = s.mem.rdN c 4 := by
    apply rdN_wr64_other hm1wr
    rcases hdisj with h | h
    · exact Or.inr (by omega)
    · exact Or.inl (by omega)
  have hs1rd : (s1.mem.rdN c 4).isSome := by
    rw [hs1mem, hm1rd]; exact hrd
  have hs1room : StackRoom s1 1 := by
    intro sp1 hsp1 i hi v
    have hsp1eq : sp1 = sp - 8 := by
      rw [hs1r4] at hsp1
      exact (Option.some.inj hsp1).symm
    have hi0 : i = 0 := by omega
    subst hsp1eq
    subst hi0
    have hh : (wr64 s.mem (sp - 16) v).isSome := by
      have h2 : (wr64 s.mem ((sp + W - 16) % W) v).isSome :=
        hroom sp hr4 1 (by decide) v
      rwa [hsp16mod] at h2
    have h3 : (wr64 m1 (sp - 16) v).isSome :=
      wr64_some_of_wr64 hm1wr hh
    have h4 : (sp - 8 + W - 8 * (0 + 1)) % W = sp - 16 := haddr2
    rw [hs1mem, h4]
    exact h3
  have hs1frame : ∀ m, wr64 s1.mem ((sp - 8 + W - 8) % W) (aRDC + 32) =
      some m → m.rdN c 4 = s1.mem.rdN c 4 := by
    intro m hm
    rw [haddr2] at hm
    rw [hs1mem] at hm ⊢
    apply rdN_wr64_other hm
    rcases hdisj with h | h
    · exact Or.inr (by omega)
    · exact Or.inl (by omega)
  -- the body
  obtain ⟨nb, s2, hrunb, hs2rip, hs2r0, hs2r6, hs2r4, hs2fr, hs2len,
      hs2memfr, hs2rd, hs2k, hs2memwr⟩ :=
    rdc_body_ok hs1rip hs1r6 hs1r15 hce hcW heW hs1rd hs1room hs1r4
      (by omega) (by omega) hs1len hs1frame
  -- the ret at aRDC + 0x24
  have hrd64 : rd64 s2.mem (sp - 8) = some (site + 5) := by
    have h1 : s2.mem.rdN (sp - 8) 8 = s1.mem.rdN (sp - 8) 8 :=
      hs2memfr (sp - 8) 8 (Or.inr (by omega))
    show (s2.mem.rdN (sp - 8) 8).map leVal = some (site + 5)
    rw [h1, hs1mem]
    have h := rd64_wr64 hm1wr
    rw [Nat.mod_eq_of_lt hsiteW] at h
    exact h
  have e2 : X86Int.step stage1Im s2 =
      .ok { s2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 } :=
    step_ret (im := stage1Im) (by rw [hs2rip]; exact hc.p295) hs2r4 hrd64
  obtain ⟨s3, hs3⟩ : ∃ t : X86Int.St, t =
      { s2.set 4 (some ((sp - 8 + 8) % W)) with rip := site + 5 } :=
    ⟨_, rfl⟩
  have hspret : (sp - 8 + 8) % W = sp := by
    rw [show sp - 8 + 8 = sp by omega, Nat.mod_eq_of_lt hspW]
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok s1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hs1, run_zero]
  have r2 : run stage1Im 1 (Out.ok s2) = Out.ok s3 := by
    rw [run_succ_ok stage1Im 0 s2, e2, ← hs3, run_zero]
  have hrun : run stage1Im (1 + nb + 1) (Out.ok s) = Out.ok s3 := by
    rw [show 1 + nb + 1 = 1 + (nb + 1) by omega, run_add stage1Im 1 _,
      r1, run_add stage1Im nb 1, hrunb, r2]
  have hm1fr : ∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x →
      m1.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  refine ⟨1 + nb + 1, s3, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs3]
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg 0 =
      (s.mem.rdN c 4).map leVal
    rw [reg_set_other (by decide)]
    rw [hs1mem, hm1rd] at hs2r0
    exact hs2r0
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg 6 = some (c + 4)
    rw [reg_set_other (by decide)]; exact hs2r6
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg 4 = some sp
    rw [reg_set_self (by rw [hs2len]; decide), hspret]
  · intro i hi0 hi2 hi4 hi6
    rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]
    rw [hs2fr i hi0 hi2 hi4 hi6]
    exact hs1fr i hi4
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).regs.length = 16
    rw [len_set]; exact hs2len
  · intro x k hdis
    rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).mem.rdN x k = s.mem.rdN x k
    rw [mem_set]
    have h1 : s2.mem.rdN x k = s1.mem.rdN x k := by
      apply hs2memfr
      rcases hdis with h | h
      · exact Or.inl (by omega)
      · exact Or.inr (by omega)
    rw [h1, hs1mem]
    apply hm1fr
    rcases hdis with h | h
    · exact Or.inl (by omega)
    · exact Or.inr h
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).mem.rdN c 4 = s.mem.rdN c 4
    rw [mem_set]
    rw [hs1mem, hm1rd] at hs2rd
    exact hs2rd
  · rw [hs3]
    show (s2.set 4 (some ((sp - 8 + 8) % W))).k = s.k
    rw [← hs1k]; exact hs2k
  · obtain ⟨m2', hm2', hs2mem'⟩ := hs2memwr
    refine ⟨m1, m2', hm1wr, ?_, ?_⟩
    · rwa [hs1mem] at hm2'
    · rw [hs3]
      show (s2.set 4 (some ((sp - 8 + 8) % W))).mem = m2'
      rw [mem_set]; exact hs2mem'

/-- A parser `call RDC` site, refusal case: fewer than four
bytes left sends the run to `aFail1` with `rsp` still decremented
by the call, and memory changed only in the return slot. -/
theorem call_rdc_fail {site c e sp : Nat} {s : X86Int.St}
    (hsite : stage1Im site = some (.call aRDC, 5))
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : ¬ c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hroom : StackRoom s 1)
    (hr4 : s.reg 4 = some sp) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = aFail1 ∧ s'.reg 4 = some (sp - 8) ∧
      (∀ i, i ≠ 2 → i ≠ 4 → s'.reg i = s.reg i) ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x → s'.mem.rdN x k = s.mem.rdN x k) ∧
      s'.regs.length = 16 ∧ s'.k = s.k := by
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W)
      (site + 5) = some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) (site + 5))
  have hm1 : wr64 s.mem ((sp + W - 8) % W) (site + 5) = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) (site + 5) = some m1 := by
    rwa [hsp8mod] at hm1
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    step_call (im := stage1Im) (s := s) (m := m1)
      (by rw [hrip]; exact hsite) hr4 (by rw [hrip]; exact hm1)
  obtain ⟨s1, hs1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := aRDC } :=
    ⟨_, rfl⟩
  have hs1rip : s1.rip = aRDC := by rw [hs1]
  have hs1r6 : s1.reg 6 = some c := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 6 = some c
    rw [reg_set_other (by decide)]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 15 = some e
    rw [reg_set_other (by decide)]; exact hr15
  have hs1r4 : s1.reg 4 = some (sp - 8) := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (by rw [hlen16]; decide), hsp8mod]
  have hs1fr : ∀ i, i ≠ 4 → s1.reg i = s.reg i := by
    intro i hi
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have hs1mem : s1.mem = m1 := by rw [hs1]
  have hs1len : s1.regs.length = 16 := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have hs1k : s1.k = s.k := by
    rw [hs1]
    show (s.set 4 (some ((sp + W - 8) % W))).k = s.k
    rfl
  obtain ⟨nb, s2, hrunb, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    rdc_body_fail hs1rip hs1r6 hs1r15 hce hcW heW hs1len
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok s1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← hs1, run_zero]
  have hrun : run stage1Im (1 + nb) (Out.ok s) = Out.ok s2 := by
    rw [run_add stage1Im 1 nb, r1, hrunb]
  refine ⟨1 + nb, s2, hrun, hs2rip, ?_, ?_, ?_, hs2len, ?_⟩
  · have h := hs2fr 4 (by decide)
    rw [h]; exact hs1r4
  · intro i hi2 hi4
    rw [hs2fr i hi2]; exact hs1fr i hi4
  · intro x k hdis
    rw [hs2mem, hs1mem]
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  · rw [← hs1k]; exact hs2k

/-- Parser header, first half: from `loaded` (`base + 0x134`,
`rbx` the file buffer, `r13` its length) to just after the
`SPR1` magic check (`base + 0x15b`), with `r15` the end pointer
and `rsi` the cursor at the buffer start.  Memory and kernel
are untouched. -/
theorem parser_header_a {B L : Nat} {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x134) (hr3 : s.reg 3 = some B)
    (hr13 : s.reg 13 = some L) (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hmagic : leVal (bs.take 4) % 2 ^ 32 = 827478099)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x15b ∧ s'.reg 15 = some (B + L) ∧
      s'.reg 6 = some B ∧
      (∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hBW : B < W := by omega
  have hbslen : bs.length = 20 := rdN_length h20
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x134: mov r15, rbx
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 15 (some B) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p134) hr3
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 15 (some B) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x137 := by
    rw [ht1]; show s.rip + 3 = base + 0x137; rw [hrip]
  have ht1r15 : t1.reg 15 = some B := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).reg 15 = some B
    rw [reg_with_rip, reg_set_self (hlen 15 (by decide))]
  have ht1r13 : t1.reg 13 = some L := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).reg 13 = some L
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr13
  have ht1fr : ∀ i, i ≠ 15 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 15 (some B) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  -- 0x137: add r15, r13
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 15 (some (aluF .add B L).1) with
        fl := (aluF .add B L).2, rip := t1.rip + 3 } :=
    step_alu_add (by rw [ht1rip]; exact hc.p137) ht1r15 ht1r13
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 15 (some (aluF .add B L).1) with
        fl := (aluF .add B L).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x13a := by
    rw [ht2]; show t1.rip + 3 = base + 0x13a; rw [ht1rip]
  have ht2r15 : t2.reg 15 = some (B + L) := by
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).reg 15 = some (B + L)
    rw [reg_set_self (by rw [ht1len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt hBL]
  have ht2fr : ∀ i, i ≠ 15 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 15 (some (aluF .add B L).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2r3 : t2.reg 3 = some B := by
    rw [ht2fr 3 (by decide)]; exact hr3
  have ht2r13 : t2.reg 13 = some L := by
    rw [ht2fr 13 (by decide)]; exact hr13
  -- 0x13a: mov rsi, rbx
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 6 (some B) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p13a) ht2r3
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 6 (some B) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x13d := by
    rw [ht3]; show t2.rip + 3 = base + 0x13d; rw [ht2rip]
  have ht3r6 : t3.reg 6 = some B := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg 6 = some B
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3r15 : t3.reg 15 = some (B + L) := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg 15 =
      some (B + L)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r15
  have ht3r13 : t3.reg 13 = some L := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg 13 = some L
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r13
  have ht3fr : ∀ i, i ≠ 6 → i ≠ 15 → t3.reg i = s.reg i := by
    intro i hi6 hi15
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi6]; exact ht2fr i hi15
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  -- 0x13d: cmp r13, 16
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub L (16 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p13d) ht3r13
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub L (16 % W)).2, rip := t3.rip + 4 } :=
    ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x141 := by
    rw [ht4]; show t3.rip + 4 = base + 0x141; rw [ht3rip]
  have h16W : 16 % W = 16 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some true := by
    have hcf : t4.fl.cf = some false := by
      rw [ht4]
      show (aluF .sub L (16 % W)).2.cf = some false
      rw [h16W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- 0x141: jae 0x148
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x148 } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p141) ht4fl
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x148 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x148 := by rw [ht5]
  have ht5r6 : t5.reg 6 = some B := by
    have h4r6 : t4.reg 6 = some B := by
      rw [ht4]; show t3.reg 6 = some B; exact ht3r6
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).reg 6 = some B
    rw [reg_with_rip]; exact h4r6
  have ht5r15 : t5.reg 15 = some (B + L) := by
    have h4r15 : t4.reg 15 = some (B + L) := by
      rw [ht4]; show t3.reg 15 = some (B + L); exact ht3r15
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).reg 15 = some (B + L)
    rw [reg_with_rip]; exact h4r15
  have ht5fr : ∀ i, i ≠ 6 → i ≠ 15 → t5.reg i = s.reg i := by
    intro i hi6 hi15
    have h4i : t4.reg i = s.reg i := by
      rw [ht4]; show t3.reg i = s.reg i; exact ht3fr i hi6 hi15
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).reg i = s.reg i
    rw [reg_with_rip]; exact h4i
  have ht5mem : t5.mem = s.mem := by
    have h4m : t4.mem = s.mem := by
      rw [ht4]; show t3.mem = s.mem; exact ht3mem
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).mem = s.mem
    rw [mem_with_rip]; exact h4m
  have ht5len : t5.regs.length = 16 := by
    have h4l : t4.regs.length = 16 := by
      rw [ht4]; show t3.regs.length = 16; exact ht3len
    rw [ht5]
    show ({ t4 with rip := base + 0x148 }).regs.length = 16
    rw [len_with_rip]; exact h4l
  -- 0x148: ld rax, [rsi]
  have hB8 : s.mem.rdN B 8 = some (bs.take 8) := by
    have h := rdN_slice h20 (show 0 + 8 ≤ 20 by decide)
    simpa using h
  have hld : rd64 t5.mem (ea B 0) = some (leVal (bs.take 8)) := by
    unfold rd64 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hBW, ht5mem, hB8]; rfl
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (leVal (bs.take 8))) with rip := t5.rip + 3 } :=
    step_ld (by rw [ht5rip]; exact hc.p148) ht5r6 hld
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (leVal (bs.take 8))) with rip := t5.rip + 3 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x14b := by
    rw [ht6]; show t5.rip + 3 = base + 0x14b; rw [ht5rip]
  have ht6r0 : t6.reg 0 = some (leVal (bs.take 8)) := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg 0 = some (leVal (bs.take 8))
    rw [reg_with_rip, reg_set_self (by rw [ht5len]; decide)]
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → t6.reg i = s.reg i := by
    intro i hi0 hi6 hi15
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht5fr i hi6 hi15
  have ht6r6 : t6.reg 6 = some B := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg 6 = some B
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht5r6
  have ht6r15 : t6.reg 15 = some (B + L) := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).reg 15 = some (B + L)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht5r15
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show ({ t5.set 0 (some (leVal (bs.take 8))) with
      rip := t5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht5len
  -- 0x14b: mov32 rax, eax
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 } :=
    step_mov32 (by rw [ht6rip]; exact hc.p14b) ht6r0
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x14d := by
    rw [ht7]; show t6.rip + 2 = base + 0x14d; rw [ht6rip]
  have htake8len : (bs.take 8).length = 8 := by
    rw [List.length_take, hbslen]; decide
  have htk : (bs.take 8).take 4 = bs.take 4 := by
    rw [List.take_take, show min 4 8 = 4 from by decide]
  have ht7r0 : t7.reg 0 = some 827478099 := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 0 = some 827478099
    rw [reg_with_rip, reg_set_self (by rw [ht6len]; decide),
      leVal_take4_mod32 htake8len, htk, hmagic]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 15 → t7.reg i = s.reg i := by
    intro i hi0 hi6 hi15
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht6fr i hi0 hi6 hi15
  have ht7r6 : t7.reg 6 = some B := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 6 = some B
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht6r6
  have ht7r15 : t7.reg 15 = some (B + L) := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).reg 15 = some (B + L)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht6r15
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
      rip := t6.rip + 2 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht6len
  -- 0x14d: cmp rax, "SPR1"
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7 with fl := (aluF .sub 827478099 (827478099 % W)).2, rip := t7.rip + 7 } :=
    step_cmpI (by rw [ht7rip]; exact hc.p14d) ht7r0
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7 with fl := (aluF .sub 827478099 (827478099 % W)).2, rip := t7.rip + 7 } :=
    ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x154 := by
    rw [ht8]; show t7.rip + 7 = base + 0x154; rw [ht7rip]
  have himmW : 827478099 % W = 827478099 :=
    Nat.mod_eq_of_lt (by unfold W; decide)
  have ht8fl : X86Int.cond t8.fl 4 = some true := by
    have hz : (aluF .sub 827478099 (827478099 % W)).1 = 0 := by
      rw [aluF_sub_fst, himmW,
        show 827478099 + W - 827478099 = W by omega, Nat.mod_self]
    have hzf : t8.fl.zf = some true := by
      rw [ht8]
      show (aluF .sub 827478099 (827478099 % W)).2.zf = some true
      simp [aluF_zf, hz]
    have hcc := cond_cc4 hzf
    simpa using hcc
  -- 0x154: je 0x15b
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8 with rip := base + 0x15b } := by
    have h := step_jcc (im := stage1Im) (s := t8)
      (by rw [ht8rip]; exact hc.p154) ht8fl
    simpa using h
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8 with rip := base + 0x15b } := ⟨_, rfl⟩
  -- compose
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have hrun : run stage1Im 9 (Out.ok s) = Out.ok t9 := by
    rw [show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r1,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r2,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r3,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r4,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r5,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r6,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r7,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r8, r9]
  have ht9k : t9.k = s.k := by
    have h7k : t7.k = s.k := by
      have h6k : t6.k = s.k := by
        have h5k : t5.k = s.k := by
          have h4k : t4.k = s.k := by
            have h3k : t3.k = s.k := by
              have h2k : t2.k = s.k := by
                rw [ht2]
                show (t1.set 15 (some (aluF .add B L).1)).k = s.k
                rw [ht1]
                show ({ s.set 15 (some B) with rip := s.rip + 3 }).k = s.k
                rfl
              rw [ht3]
              show ({ t2.set 6 (some B) with rip := t2.rip + 3 }).k = s.k
              exact h2k
            rw [ht4]
            show t3.k = s.k
            exact h3k
          rw [ht5]
          show ({ t4 with rip := base + 0x148 }).k = s.k
          exact h4k
        rw [ht6]
        show ({ t5.set 0 (some (leVal (bs.take 8))) with
          rip := t5.rip + 3 }).k = s.k
        exact h5k
      rw [ht7]
      show ({ t6.set 0 (some (leVal (bs.take 8) % 2 ^ 32)) with
        rip := t6.rip + 2 }).k = s.k
      exact h6k
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).k = s.k
    rw [ht8]
    show t7.k = s.k
    exact h7k
  refine ⟨9, t9, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ht9k⟩
  · rw [ht9]
  · have h8r15 : t8.reg 15 = some (B + L) := by
      rw [ht8]; show t7.reg 15 = some (B + L); exact ht7r15
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).reg 15 = some (B + L)
    rw [reg_with_rip]; exact h8r15
  · have h8r6 : t8.reg 6 = some B := by
      rw [ht8]; show t7.reg 6 = some B; exact ht7r6
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).reg 6 = some B
    rw [reg_with_rip]; exact h8r6
  · intro i hi0 hi6 hi15
    have h8i : t8.reg i = s.reg i := by
      rw [ht8]; show t7.reg i = s.reg i; exact ht7fr i hi0 hi6 hi15
    rw [ht9]
    show ({ t8 with rip := base + 0x15b }).reg i = s.reg i
    rw [reg_with_rip]; exact h8i
  · rw [ht9]
    show ({ t8 with rip := base + 0x15b }).mem = s.mem
    rw [mem_with_rip, ht8]
    show t7.mem = s.mem
    exact ht7mem
  · rw [ht9]
    show ({ t8 with rip := base + 0x15b }).regs.length = 16
    rw [len_with_rip, ht8]
    show t7.regs.length = 16
    exact ht7len

/-- One header field load: `mov rdx, rsi; add rdx, off;
ld rax, [rdx]; mov32 dst, eax`, leaving in `dst` the `u32` at
`B + off` (as computed: the low 32 bits of the 8-byte load). -/
theorem field_load {site B off dst len4 : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hc1 : stage1Im site = some (.mov 2 6, 3))
    (hc2 : stage1Im (site + 3) = some (.aluI .add 2 off, 4))
    (hc3 : stage1Im (site + 7) = some (.ld 0 2 0, 3))
    (hc4 : stage1Im (site + 10) = some (.mov32 dst 0, len4))
    (hrip : s.rip = site) (hr6 : s.reg 6 = some B)
    (h20 : s.mem.rdN B 20 = some bs) (hoff : off + 8 ≤ 20)
    (hoffW : off < W) (hBoff : B + off < W)
    (hdst : dst < 16) (hdst6 : dst ≠ 6)
    (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 4 (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 10 + len4 ∧
      s'.reg dst = some (leVal ((bs.drop off).take 4) % 2 ^ 32) ∧
      s'.reg 6 = some B ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ dst → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hbslen : bs.length = 20 := rdN_length h20
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- mov rdx, rsi
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some B) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc1) hr6
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some B) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = site + 3 := by
    rw [ht1]; show s.rip + 3 = site + 3; rw [hrip]
  have ht1r2 : t1.reg 2 = some B := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).reg 2 = some B
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some B := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).reg 6 = some B
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some B) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  -- add rdx, off
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .add B (off % W)).1) with
        fl := (aluF .add B (off % W)).2, rip := t1.rip + 4 } :=
    step_aluI_add (by rw [ht1rip]; exact hc2) ht1r2
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .add B (off % W)).1) with
        fl := (aluF .add B (off % W)).2, rip := t1.rip + 4 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = site + 7 := by
    rw [ht2]; show t1.rip + 4 = site + 7; rw [ht1rip]
  have hoffm : off % W = off := Nat.mod_eq_of_lt hoffW
  have ht2r2 : t2.reg 2 = some (B + off) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).reg 2 = some (B + off)
    rw [reg_set_self (by rw [ht1len]; decide), aluF_add_fst, hoffm,
      Nat.mod_eq_of_lt hBoff]
  have ht2r6 : t2.reg 6 = some B := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).reg 6 = some B
    rw [reg_set_other (by decide)]; exact ht1r6
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .add B (off % W)).1)).regs.length = 16
    rw [len_set]; exact ht1len
  -- ld rax, [rdx]
  have hslice : s.mem.rdN (B + off) 8 = some ((bs.drop off).take 8) :=
    rdN_slice h20 hoff
  have hld : rd64 t2.mem (ea (B + off) 0) =
      some (leVal ((bs.drop off).take 8)) := by
    unfold rd64 ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hBoff, ht2mem, hslice]; rfl
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 0 (some (leVal ((bs.drop off).take 8))) with
        rip := t2.rip + 3 } :=
    step_ld (by rw [ht2rip]; exact hc3) ht2r2 hld
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 0 (some (leVal ((bs.drop off).take 8))) with
        rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = site + 10 := by
    rw [ht3]; show t2.rip + 3 = site + 10; rw [ht2rip]
  have ht3r0 : t3.reg 0 = some (leVal ((bs.drop off).take 8)) := by
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).reg 0 = some (leVal ((bs.drop off).take 8))
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht2fr i hi2
  have ht3r6 : t3.reg 6 = some B := by
    rw [ht3fr 6 (by decide) (by decide)]; exact hr6
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
      rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  -- mov32 dst, eax
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
        rip := t3.rip + len4 } :=
    step_mov32 (by rw [ht3rip]; exact hc4) ht3r0
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
        rip := t3.rip + len4 } := ⟨_, rfl⟩
  have htk8len : ((bs.drop off).take 8).length = 8 := by
    rw [List.length_take, List.length_drop, hbslen,
      Nat.min_eq_left (by omega)]
  have htk : ((bs.drop off).take 8).take 4 = (bs.drop off).take 4 := by
    rw [List.take_take, show min 4 8 = 4 from by decide]
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have hrun : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
    rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
  have ht4k : t4.k = s.k := by
    have h3k : t3.k = s.k := by
      have h2k : t2.k = s.k := by
        rw [ht2]
        show (t1.set 2 (some (aluF .add B (off % W)).1)).k = s.k
        rw [ht1]
        show ({ s.set 2 (some B) with rip := s.rip + 3 }).k = s.k
        rfl
      rw [ht3]
      show ({ t2.set 0 (some (leVal ((bs.drop off).take 8))) with
        rip := t2.rip + 3 }).k = s.k
      exact h2k
    rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).k = s.k
    exact h3k
  refine ⟨t4, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ht4k⟩
  · rw [ht4]; show t3.rip + len4 = site + 10 + len4; rw [ht3rip]
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).reg dst =
      some (leVal ((bs.drop off).take 4) % 2 ^ 32)
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; exact hdst),
      leVal_take4_mod32 htk8len, htk]
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).reg 6 = some B
    rw [reg_with_rip,
      reg_set_other (show (6 : Nat) ≠ dst from fun h => hdst6 h.symm)]
    exact ht3r6
  · intro i hi0 hi2 hid
    rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hid]; exact ht3fr i hi0 hi2
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  · rw [ht4]
    show ({ t3.set dst (some (leVal ((bs.drop off).take 8) % 2 ^ 32)) with
      rip := t3.rip + len4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len

/-- Parser header, second half: from `base + 0x15b` (cursor at
the buffer start) to the data-skip loop head (`base + 0x18f`),
loading `main` into `r12`, `nwords` into `r10`, `ndata` into
`rcx`, advancing the cursor past the 16-byte header, and
refusing zero `nwords` (the refusal case is separate). -/
theorem parser_header_b {B L : Nat} {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x15b) (hr6 : s.reg 6 = some B)
    (hBL : B + L < W) (hL16 : 16 ≤ L)
    (h20 : s.mem.rdN B 20 = some bs)
    (hnw : leVal ((bs.drop 8).take 4) % 2 ^ 32 ≠ 0)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧
      s'.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) ∧
      s'.reg 10 = some (leVal ((bs.drop 8).take 4) % 2 ^ 32) ∧
      s'.reg 1 = some (leVal ((bs.drop 12).take 4) % 2 ^ 32) ∧
      s'.reg 6 = some (B + 16) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
        s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hBoff : ∀ o : Nat, o ≤ 12 → B + o < W := by
    intro o ho; omega
  -- main := u32 at B+4 (site 0x15b, exit 0x168)
  obtain ⟨s1, hrun1, hs1rip, hs1r12, hs1r6, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    field_load (site := base + 0x15b) (off := 4) (dst := 12) (len4 := 3)
      (by exact hc.p15b)
      (by exact hc.p15e) (by exact hc.p162) (by exact hc.p165)
      hrip hr6 h20 (by decide) (by unfold W; decide) (hBoff 4 (by decide))
      (by decide) (by decide) hlen16
  -- nwords := u32 at B+8 (site 0x168, exit 0x175)
  have hs1rip' : s1.rip = base + 0x168 := by
    rw [hs1rip]
  have h20_1 : s1.mem.rdN B 20 = some bs := by rw [hs1mem]; exact h20
  obtain ⟨s2, hrun2, hs2rip, hs2r10, hs2r6, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    field_load (site := base + 0x168) (off := 8) (dst := 10) (len4 := 3)
      (by exact hc.p168) (by exact hc.p16b) (by exact hc.p16f)
      (by exact hc.p172)
      hs1rip' hs1r6 h20_1 (by decide) (by unfold W; decide)
      (hBoff 8 (by decide)) (by decide) (by decide) hs1len
  -- ndata := u32 at B+12 (site 0x175, exit 0x181)
  have hs2rip' : s2.rip = base + 0x175 := by
    rw [hs2rip]
  have h20_2 : s2.mem.rdN B 20 = some bs := by rw [hs2mem]; exact h20_1
  obtain ⟨s3, hrun3, hs3rip, hs3r1, hs3r6, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    field_load (site := base + 0x175) (off := 12) (dst := 1) (len4 := 2)
      (by exact hc.p175) (by exact hc.p178) (by exact hc.p17c)
      (by exact hc.p17f)
      hs2rip' hs2r6 h20_2 (by decide) (by unfold W; decide)
      (hBoff 12 (by decide)) (by decide) (by decide) hs2len
  have hs3rip' : s3.rip = base + 0x181 := by
    rw [hs3rip]
  have hs3r12 : s3.reg 12 =
      some (leVal ((bs.drop 4).take 4) % 2 ^ 32) := by
    have h2 : s2.reg 12 = some (leVal ((bs.drop 4).take 4) % 2 ^ 32) := by
      rw [hs2fr 12 (by decide) (by decide) (by decide)]; exact hs1r12
    rw [hs3fr 12 (by decide) (by decide) (by decide)]; exact h2
  have hs3r10 : s3.reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
    rw [hs3fr 10 (by decide) (by decide) (by decide)]; exact hs2r10
  -- 0x181: add rsi, 16
  have e10 : X86Int.step stage1Im s3 =
      .ok { s3.set 6 (some (aluF .add B (16 % W)).1) with
        fl := (aluF .add B (16 % W)).2, rip := s3.rip + 4 } :=
    step_aluI_add (by rw [hs3rip']; exact hc.p181) hs3r6
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s3.set 6 (some (aluF .add B (16 % W)).1) with
        fl := (aluF .add B (16 % W)).2, rip := s3.rip + 4 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x185 := by
    rw [ht1]; show s3.rip + 4 = base + 0x185; rw [hs3rip']
  have h16W : 16 % W = 16 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht1r6 : t1.reg 6 = some (B + 16) := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).reg 6 = some (B + 16)
    rw [reg_set_self (by rw [hs3len]; decide), aluF_add_fst, h16W,
      Nat.mod_eq_of_lt (by omega)]
  have ht1r10 : t1.reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32) := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).reg 10 =
      some (leVal ((bs.drop 8).take 4) % 2 ^ 32)
    rw [reg_set_other (by decide)]; exact hs3r10
  have ht1fr : ∀ i, i ≠ 6 → t1.reg i = s3.reg i := by
    intro i hi
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).reg i = s3.reg i
    rw [reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).mem = s.mem
    rw [mem_set, hs3mem, hs2mem, hs1mem]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show (s3.set 6 (some (aluF .add B (16 % W)).1)).regs.length = 16
    rw [len_set]; exact hs3len
  -- 0x185: test r10, r10
  have e11 : X86Int.step stage1Im t1 =
      .ok { t1 with
        fl := (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
          (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2,
        rip := t1.rip + 3 } :=
    step_test (by rw [ht1rip]; exact hc.p185) ht1r10 ht1r10
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with
        fl := (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
          (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2,
        rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x188 := by
    rw [ht2]; show t1.rip + 3 = base + 0x188; rw [ht1rip]
  have ht2fl : X86Int.cond t2.fl 5 = some true := by
    have hzf : t2.fl.zf = some false := by
      rw [ht2]
      show (aluF .and (leVal ((bs.drop 8).take 4) % 2 ^ 32)
        (leVal ((bs.drop 8).take 4) % 2 ^ 32)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [hnw]
    have hcc := cond_cc5 hzf
    simpa using hcc
  -- 0x188: jne 0x18f
  have e12 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x18f } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p188) ht2fl
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x18f } := ⟨_, rfl⟩
  -- compose
  have r10 : run stage1Im 1 (Out.ok s3) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s3, e10, ← ht1, run_zero]
  have r11 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e11, ← ht2, run_zero]
  have r12 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e12, ← ht3, run_zero]
  have hrun : run stage1Im 15 (Out.ok s) = Out.ok t3 := by
    rw [show (15 : Nat) = 4 + 11 by omega, run_add stage1Im 4 11, hrun1,
      show (11 : Nat) = 4 + 7 by omega, run_add stage1Im 4 7, hrun2,
      show (7 : Nat) = 4 + 3 by omega, run_add stage1Im 4 3, hrun3,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r10,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r11, r12]
  have ht3k : t3.k = s.k := by
    have h2k : t2.k = s.k := by
      have h1k : t1.k = s.k := by
        rw [ht1]
        show (s3.set 6 (some (aluF .add B (16 % W)).1)).k = s.k
        exact hs3k.trans (hs2k.trans hs1k)
      rw [ht2]
      show t1.k = s.k
      exact h1k
    rw [ht3]
    show ({ t2 with rip := base + 0x18f }).k = s.k
    exact h2k
  have ht3fr' : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 → i ≠ 12 →
      t3.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi6 hi10 hi12
    have h2i : t2.reg i = s.reg i := by
      have h1i : t1.reg i = s.reg i := by
        rw [ht1fr i hi6]
        rw [hs3fr i hi0 hi2 hi1]
        rw [hs2fr i hi0 hi2 hi10]
        exact hs1fr i hi0 hi2 hi12
      rw [ht2]
      show t1.reg i = s.reg i
      exact h1i
    rw [ht3]
    show ({ t2 with rip := base + 0x18f }).reg i = s.reg i
    rw [reg_with_rip]; exact h2i
  have ht3r : ∀ j : Nat, j = 12 ∨ j = 10 ∨ j = 1 ∨ j = 6 →
      t3.reg j = t1.reg j := by
    intro j _
    have h2j : t2.reg j = t1.reg j := by
      rw [ht2]; show t1.reg j = t1.reg j; rfl
    rw [ht3]
    show ({ t2 with rip := base + 0x18f }).reg j = t1.reg j
    rw [reg_with_rip]; exact h2j
  refine ⟨15, t3, hrun, ?_, ?_, ?_, ?_, ?_, ht3fr', ?_, ?_, ht3k⟩
  · rw [ht3]
  · rw [ht3r 12 (Or.inl rfl), ht1fr 12 (by decide)]; exact hs3r12
  · rw [ht3r 10 (Or.inr (Or.inl rfl))]; exact ht1r10
  · rw [ht3r 1 (Or.inr (Or.inr (Or.inl rfl))), ht1fr 1 (by decide)]
    exact hs3r1
  · rw [ht3r 6 (Or.inr (Or.inr (Or.inr rfl)))]; exact ht1r6
  · rw [ht3]
    show ({ t2 with rip := base + 0x18f }).mem = s.mem
    rw [mem_with_rip, ht2]
    show t1.mem = s.mem
    exact ht1mem
  · rw [ht3]
    show ({ t2 with rip := base + 0x18f }).regs.length = 16
    rw [len_with_rip, ht2]
    show t1.regs.length = 16
    exact ht1len

/-- The data segments at cursor `c` in memory: lengths `ds`,
each stored as a `u32` at the cursor followed by that many
bytes, ending at cursor `c'`.  Mirrors `decMany decData`. -/
inductive DataSeq (m : Mem) : Nat → List Nat → Nat → Prop
  | nil (c : Nat) : DataSeq m c [] c
  | cons {c d : Nat} {ds : List Nat} {c' : Nat}
      (hd : rd32 m c = some d) (hrest : DataSeq m (c + 4 + d) ds c') :
      DataSeq m c (d :: ds) c'

theorem DataSeq.le {m : Mem} {c : Nat} {ds : List Nat} {c' : Nat}
    (h : DataSeq m c ds c') : c ≤ c' := by
  induction h with
  | nil c => exact Nat.le_refl c
  | cons hd hrest ih => omega

/-- One data-loop iteration, success case: from the loop head
`base + 0x18f` back to it, consuming one length-prefixed
segment. -/
theorem parser_data_step_ok {c e d n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x18f) (hr1 : s.reg 1 = some n)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hn : 0 < n) (hnW : n < W) (hce : c + 4 ≤ e)
    (hd : rd32 s.mem c = some d) (hcd : c + 4 + d ≤ e)
    (heW : e < W) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 16 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x18f ∧ s'.reg 1 = some (n - 1) ∧
      s'.reg 6 = some (c + 4 + d) ∧ s'.reg 15 = some e ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hcW : c < W := by omega
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x18f: test rcx, rcx
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and n n).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p18f) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and n n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x192 := by
    rw [ht1]; show s.rip + 3 = base + 0x192; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 4 = some false := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .and n n).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [show n ≠ 0 by omega]
    have hcc := cond_cc4 hzf
    simpa using hcc
  -- 0x192: jz not taken
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x194 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p192) ht1fl
    simpa [ht1rip] using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x194 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x194 := by rw [ht2]
  have ht2r15 : t2.reg 15 = some e := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).reg 15 = some e
    rw [reg_with_rip, ht1]
    show s.reg 15 = some e
    exact hr15
  have ht2r6 : t2.reg 6 = some c := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).reg 6 = some c
    rw [reg_with_rip, ht1]
    show s.reg 6 = some c
    exact hr6
  have ht2fr : ∀ i, t2.reg i = s.reg i := by
    intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    rfl
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).mem = s.mem
    rw [mem_with_rip, ht1]
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1 with rip := base + 0x194 }).regs.length = 16
    rw [len_with_rip, ht1]
    show s.regs.length = 16
    exact hlen16
  -- 0x194: mov rdx, r15
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 2 (some e) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p194) ht2r15
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 2 (some e) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x197 := by
    rw [ht3]; show t2.rip + 3 = base + 0x197; rw [ht2rip]
  have ht3r2 : t3.reg 2 = some e := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3r6 : t3.reg 6 = some c := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r6
  have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht2fr i
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 2 (some e) with rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  -- 0x197: sub rdx, rsi
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t3.rip + 3 } :=
    step_alu_sub (by rw [ht3rip]; exact hc.p197) ht3r2 ht3r6
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x19a := by
    rw [ht4]; show t3.rip + 3 = base + 0x19a; rw [ht3rip]
  have hsub1 : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht4r2 : t4.reg 2 = some (e - c) := by
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht3len]; decide), hsub1]
  have ht4fr : ∀ i, i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht3fr i hi
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show (t3.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht3len
  -- 0x19a: cmp rdx, 4
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t4.rip + 4 } :=
    step_cmpI (by rw [ht4rip]; exact hc.p19a) ht4r2
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with fl := (aluF .sub (e - c) (4 % W)).2, rip := t4.rip + 4 } :=
    ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x19e := by
    rw [ht5]; show t4.rip + 4 = base + 0x19e; rw [ht4rip]
  have h4W : 4 % W = 4 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht5fl : X86Int.cond t5.fl 3 = some true := by
    have hcf : t5.fl.cf = some false := by
      rw [ht5]
      show (aluF .sub (e - c) (4 % W)).2.cf = some false
      rw [h4W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- 0x19e: jae 0x1a5
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5 with rip := base + 0x1a5 } := by
    have h := step_jcc (im := stage1Im) (s := t5)
      (by rw [ht5rip]; exact hc.p19e) ht5fl
    simpa using h
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5 with rip := base + 0x1a5 } := ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x1a5 := by rw [ht6]
  have ht6r6 : t6.reg 6 = some c := by
    have h5r6 : t5.reg 6 = some c := by
      rw [ht5]; show t4.reg 6 = some c
      rw [ht4fr 6 (by decide)]; exact hr6
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).reg 6 = some c
    rw [reg_with_rip]; exact h5r6
  have ht6fr : ∀ i, i ≠ 2 → t6.reg i = s.reg i := by
    intro i hi
    have h5i : t5.reg i = s.reg i := by
      rw [ht5]; show t4.reg i = s.reg i; exact ht4fr i hi
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).reg i = s.reg i
    rw [reg_with_rip]; exact h5i
  have ht6mem : t6.mem = s.mem := by
    have h5m : t5.mem = s.mem := by
      rw [ht5]; show t4.mem = s.mem; exact ht4mem
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).mem = s.mem
    rw [mem_with_rip]; exact h5m
  have ht6len : t6.regs.length = 16 := by
    have h5l : t5.regs.length = 16 := by
      rw [ht5]; show t4.regs.length = 16; exact ht4len
    rw [ht6]
    show ({ t5 with rip := base + 0x1a5 }).regs.length = 16
    rw [len_with_rip]; exact h5l
  -- 0x1a5: ld32 rax, [rsi]
  have hld : rd32 t6.mem (ea c 0) = some d := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hcW, ht6mem]; exact hd
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 0 (some d) with rip := t6.rip + 2 } :=
    step_ld32 (by rw [ht6rip]; exact hc.p1a5) ht6r6 hld
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 0 (some d) with rip := t6.rip + 2 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x1a7 := by
    rw [ht7]; show t6.rip + 2 = base + 0x1a7; rw [ht6rip]
  have ht7r0 : t7.reg 0 = some d := by
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).reg 0 = some d
    rw [reg_with_rip, reg_set_self (by rw [ht6len]; decide)]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → t7.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht6fr i hi2
  have ht7r6 : t7.reg 6 = some c := by
    rw [ht7fr 6 (by decide) (by decide)]; exact hr6
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show ({ t6.set 0 (some d) with rip := t6.rip + 2 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht6len
  -- 0x1a7: mov rax, rax
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 0 (some d) with rip := t7.rip + 3 } :=
    step_mov (by rw [ht7rip]; exact hc.p1a7) ht7r0
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 0 (some d) with rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x1aa := by
    rw [ht8]; show t7.rip + 3 = base + 0x1aa; rw [ht7rip]
  have ht8r0 : t8.reg 0 = some d := by
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).reg 0 = some d
    rw [reg_with_rip, reg_set_self (by rw [ht7len]; decide)]
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → t8.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht7fr i hi0 hi2
  have ht8r6 : t8.reg 6 = some c := by
    rw [ht8fr 6 (by decide) (by decide)]; exact hr6
  have ht8r15 : t8.reg 15 = some e := by
    rw [ht8fr 15 (by decide) (by decide)]; exact hr15
  have ht8mem : t8.mem = s.mem := by
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show ({ t7.set 0 (some d) with rip := t7.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht7len
  -- 0x1aa: add rsi, 4
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t8.rip + 4 } :=
    step_aluI_add (by rw [ht8rip]; exact hc.p1aa) ht8r6
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 6 (some (aluF .add c (4 % W)).1) with
        fl := (aluF .add c (4 % W)).2, rip := t8.rip + 4 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x1ae := by
    rw [ht9]; show t8.rip + 4 = base + 0x1ae; rw [ht8rip]
  have ht9r6 : t9.reg 6 = some (c + 4) := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg 6 = some (c + 4)
    rw [reg_set_self (by rw [ht8len]; decide), aluF_add_fst, h4W,
      Nat.mod_eq_of_lt (by omega)]
  have ht9r0 : t9.reg 0 = some d := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg 0 = some d
    rw [reg_set_other (by decide)]; exact ht8r0
  have ht9r15 : t9.reg 15 = some e := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg 15 = some e
    rw [reg_set_other (by decide)]; exact ht8r15
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t9.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi6]; exact ht8fr i hi0 hi2
  have ht9mem : t9.mem = s.mem := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).mem = s.mem
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c (4 % W)).1)).regs.length = 16
    rw [len_set]; exact ht8len
  -- 0x1ae: mov rdx, r15
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 2 (some e) with rip := t9.rip + 3 } :=
    step_mov (by rw [ht9rip]; exact hc.p1ae) ht9r15
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 2 (some e) with rip := t9.rip + 3 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x1b1 := by
    rw [ht10]; show t9.rip + 3 = base + 0x1b1; rw [ht9rip]
  have ht10r2 : t10.reg 2 = some e := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht9len]; decide)]
  have ht10r6 : t10.reg 6 = some (c + 4) := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg 6 =
      some (c + 4)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht9r6
  have ht10r0 : t10.reg 0 = some d := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg 0 = some d
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht9r0
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t10.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht9fr i hi0 hi2 hi6
  have ht10mem : t10.mem = s.mem := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht9len
  -- 0x1b1: sub rdx, rsi
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10.set 2 (some (aluF .sub e (c + 4)).1) with
        fl := (aluF .sub e (c + 4)).2, rip := t10.rip + 3 } :=
    step_alu_sub (by rw [ht10rip]; exact hc.p1b1) ht10r2 ht10r6
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10.set 2 (some (aluF .sub e (c + 4)).1) with
        fl := (aluF .sub e (c + 4)).2, rip := t10.rip + 3 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x1b4 := by
    rw [ht11]; show t10.rip + 3 = base + 0x1b4; rw [ht10rip]
  have hsub2 : (aluF .sub e (c + 4)).1 = e - (c + 4) := by
    rw [aluF_sub_fst, show e + W - (c + 4) = (e - (c + 4)) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht11r2 : t11.reg 2 = some (e - (c + 4)) := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg 2 =
      some (e - (c + 4))
    rw [reg_set_self (by rw [ht10len]; decide), hsub2]
  have ht11r0 : t11.reg 0 = some d := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg 0 = some d
    rw [reg_set_other (by decide)]; exact ht10r0
  have ht11r6 : t11.reg 6 = some (c + 4) := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg 6 = some (c + 4)
    rw [reg_set_other (by decide)]; exact ht10r6
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t11.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht10fr i hi0 hi2 hi6
  have ht11mem : t11.mem = s.mem := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).mem = s.mem
    rw [mem_set]; exact ht10mem
  have ht11len : t11.regs.length = 16 := by
    rw [ht11]
    show (t10.set 2 (some (aluF .sub e (c + 4)).1)).regs.length = 16
    rw [len_set]; exact ht10len
  -- 0x1b4: sub rdx, rax
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (aluF .sub (e - (c + 4)) d).1) with
        fl := (aluF .sub (e - (c + 4)) d).2, rip := t11.rip + 3 } :=
    step_alu_sub (by rw [ht11rip]; exact hc.p1b4) ht11r2 ht11r0
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (aluF .sub (e - (c + 4)) d).1) with
        fl := (aluF .sub (e - (c + 4)) d).2, rip := t11.rip + 3 } :=
    ⟨_, rfl⟩
  have ht12rip : t12.rip = base + 0x1b7 := by
    rw [ht12]; show t11.rip + 3 = base + 0x1b7; rw [ht11rip]
  have ht12fl : X86Int.cond t12.fl 3 = some true := by
    have hcf : t12.fl.cf = some false := by
      rw [ht12]
      show (aluF .sub (e - (c + 4)) d).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht12fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t12.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht11fr i hi0 hi2 hi6
  have ht12r6 : t12.reg 6 = some (c + 4) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).reg 6 =
      some (c + 4)
    rw [reg_set_other (by decide)]; exact ht11r6
  have ht12r0 : t12.reg 0 = some d := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).reg 0 = some d
    rw [reg_set_other (by decide)]; exact ht11r0
  have ht12mem : t12.mem = s.mem := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).mem = s.mem
    rw [mem_set]; exact ht11mem
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).regs.length = 16
    rw [len_set]; exact ht11len
  -- 0x1b7: jae 0x1be
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12 with rip := base + 0x1be } := by
    have h := step_jcc (im := stage1Im) (s := t12)
      (by rw [ht12rip]; exact hc.p1b7) ht12fl
    simpa using h
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12 with rip := base + 0x1be } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x1be := by rw [ht13]
  have ht13r6 : t13.reg 6 = some (c + 4) := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).reg 6 = some (c + 4)
    rw [reg_with_rip]; exact ht12r6
  have ht13r0 : t13.reg 0 = some d := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).reg 0 = some d
    rw [reg_with_rip]; exact ht12r0
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t13.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).reg i = s.reg i
    rw [reg_with_rip]; exact ht12fr i hi0 hi2 hi6
  have ht13mem : t13.mem = s.mem := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).mem = s.mem
    rw [mem_with_rip]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12 with rip := base + 0x1be }).regs.length = 16
    rw [len_with_rip]; exact ht12len
  -- 0x1be: add rsi, rax
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13.set 6 (some (aluF .add (c + 4) d).1) with
        fl := (aluF .add (c + 4) d).2, rip := t13.rip + 3 } :=
    step_alu_add (by rw [ht13rip]; exact hc.p1be) ht13r6 ht13r0
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13.set 6 (some (aluF .add (c + 4) d).1) with
        fl := (aluF .add (c + 4) d).2, rip := t13.rip + 3 } := ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x1c1 := by
    rw [ht14]; show t13.rip + 3 = base + 0x1c1; rw [ht13rip]
  have ht14r6 : t14.reg 6 = some (c + 4 + d) := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).reg 6 =
      some (c + 4 + d)
    rw [reg_set_self (by rw [ht13len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega)]
  have ht14r1 : t14.reg 1 = some n := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).reg 1 = some n
    rw [reg_set_other (by decide), ht13fr 1 (by decide) (by decide)
      (by decide)]
    exact hr1
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t14.reg i = s.reg i := by
    intro i hi0 hi2 hi6
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).reg i = s.reg i
    rw [reg_set_other hi6]; exact ht13fr i hi0 hi2 hi6
  have ht14mem : t14.mem = s.mem := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).mem = s.mem
    rw [mem_set]; exact ht13mem
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show (t13.set 6 (some (aluF .add (c + 4) d).1)).regs.length = 16
    rw [len_set]; exact ht13len
  -- 0x1c1: sub rcx, 1
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14.set 1 (some (aluF .sub n (1 % W)).1) with
        fl := (aluF .sub n (1 % W)).2, rip := t14.rip + 4 } :=
    step_aluI_sub (by rw [ht14rip]; exact hc.p1c1) ht14r1
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14.set 1 (some (aluF .sub n (1 % W)).1) with
        fl := (aluF .sub n (1 % W)).2, rip := t14.rip + 4 } := ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x1c5 := by
    rw [ht15]; show t14.rip + 4 = base + 0x1c5; rw [ht14rip]
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht15r1 : t15.reg 1 = some (n - 1) := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).reg 1 = some (n - 1)
    rw [reg_set_self (by rw [ht14len]; decide), aluF_sub_fst, h1W,
      show n + W - 1 = (n - 1) + W by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
  have ht15fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 →
      t15.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi6
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi1]; exact ht14fr i hi0 hi2 hi6
  have ht15r6 : t15.reg 6 = some (c + 4 + d) := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).reg 6 =
      some (c + 4 + d)
    rw [reg_set_other (by decide)]; exact ht14r6
  have ht15r15 : t15.reg 15 = some e := by
    rw [ht15fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have ht15mem : t15.mem = s.mem := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).mem = s.mem
    rw [mem_set]; exact ht14mem
  have ht15len : t15.regs.length = 16 := by
    rw [ht15]
    show (t14.set 1 (some (aluF .sub n (1 % W)).1)).regs.length = 16
    rw [len_set]; exact ht14len
  -- 0x1c5: jmp 0x18f
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15 with rip := base + 0x18f } :=
    step_jmp (by rw [ht15rip]; exact hc.p1c5)
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15 with rip := base + 0x18f } := ⟨_, rfl⟩
  -- compose (16 steps)
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  have r16 : run stage1Im 1 (Out.ok t15) = Out.ok t16 := by
    rw [run_succ_ok stage1Im 0 t15, e16, ← ht16, run_zero]
  have hrun : run stage1Im 16 (Out.ok s) = Out.ok t16 := by
    rw [show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r1,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r2,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r3,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r4,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r5,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r6,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r7,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r8,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r9,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r10,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r11,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r12,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r13,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r14,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r15, r16]
  have ht16k : t16.k = s.k := by
    have h15k : t15.k = s.k := by
      have h14k : t14.k = s.k := by
        have h13k : t13.k = s.k := by
          have h12k : t12.k = s.k := by
            have h11k : t11.k = s.k := by
              have h10k : t10.k = s.k := by
                have h9k : t9.k = s.k := by
                  have h8k : t8.k = s.k := by
                    have h7k : t7.k = s.k := by
                      have h6k : t6.k = s.k := by
                        have h5k : t5.k = s.k := by
                          have h4k : t4.k = s.k := by
                            have h3k : t3.k = s.k := by
                              have h2k : t2.k = s.k := by
                                rw [ht2]
                                show ({ t1 with rip := base + 0x194 }).k =
                                  s.k
                                rw [ht1]
                              rw [ht3]
                              show ({ t2.set 2 (some e) with
                                rip := t2.rip + 3 }).k = s.k
                              exact h2k
                            rw [ht4]
                            show (t3.set 2
                              (some (aluF .sub e c).1)).k = s.k
                            exact h3k
                          rw [ht5]
                          show t4.k = s.k
                          exact h4k
                        rw [ht6]
                        show ({ t5 with rip := base + 0x1a5 }).k = s.k
                        exact h5k
                      rw [ht7]
                      show ({ t6.set 0 (some d) with
                        rip := t6.rip + 2 }).k = s.k
                      exact h6k
                    rw [ht8]
                    show ({ t7.set 0 (some d) with
                      rip := t7.rip + 3 }).k = s.k
                    exact h7k
                  rw [ht9]
                  show (t8.set 6 (some (aluF .add c (4 % W)).1)).k = s.k
                  exact h8k
                rw [ht10]
                show ({ t9.set 2 (some e) with rip := t9.rip + 3 }).k =
                  s.k
                exact h9k
              rw [ht11]
              show (t10.set 2 (some (aluF .sub e (c + 4)).1)).k = s.k
              exact h10k
            rw [ht12]
            show (t11.set 2 (some (aluF .sub (e - (c + 4)) d).1)).k = s.k
            exact h11k
          rw [ht13]
          show ({ t12 with rip := base + 0x1be }).k = s.k
          exact h12k
        rw [ht14]
        show (t13.set 6 (some (aluF .add (c + 4) d).1)).k = s.k
        exact h13k
      rw [ht15]
      show (t14.set 1 (some (aluF .sub n (1 % W)).1)).k = s.k
      exact h14k
    rw [ht16]
    show ({ t15 with rip := base + 0x18f }).k = s.k
    exact h15k
  refine ⟨t16, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ht16k⟩
  · rw [ht16]
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg 1 = some (n - 1)
    rw [reg_with_rip]; exact ht15r1
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg 6 = some (c + 4 + d)
    rw [reg_with_rip]; exact ht15r6
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg 15 = some e
    rw [reg_with_rip]; exact ht15r15
  · intro i hi0 hi1 hi2 hi6
    rw [ht16]
    show ({ t15 with rip := base + 0x18f }).reg i = s.reg i
    rw [reg_with_rip]; exact ht15fr i hi0 hi1 hi2 hi6
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).mem = s.mem
    rw [mem_with_rip]; exact ht15mem
  · rw [ht16]
    show ({ t15 with rip := base + 0x18f }).regs.length = 16
    rw [len_with_rip]; exact ht15len

/-- The whole data-skip loop, success case: from the loop head
with `ds.length` segments described by `DataSeq` to
`base + 0x1c7` (`data_done`), cursor at the end of the last
segment. -/
theorem parser_data_loop_ok {e : Nat} :
    ∀ {c : Nat} {ds : List Nat} {c' : Nat} {s : X86Int.St},
      DataSeq s.mem c ds c' → s.rip = base + 0x18f →
      s.reg 1 = some ds.length → s.reg 6 = some c →
      s.reg 15 = some e → c' ≤ e → e < W → ds.length < W →
      s.regs.length = 16 →
      ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
        s'.rip = base + 0x1c7 ∧ s'.reg 1 = some 0 ∧
        s'.reg 6 = some c' ∧ s'.reg 15 = some e ∧
        (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
        s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  intro c ds
  induction ds generalizing c with
  | nil =>
    intro c' s hseq hrip hr1 hr6 hr15 hc'e heW hlenW hlen16
    cases hseq
    have hc := stage1_parserCode
    have hr1' : s.reg 1 = some 0 := hr1
    -- test rcx, rcx (zero)
    have e1 : X86Int.step stage1Im s =
        .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
      step_test (by rw [hrip]; exact hc.p18f) hr1' hr1'
    obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
        { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
    have ht1rip : t1.rip = base + 0x192 := by
      rw [ht1]; show s.rip + 3 = base + 0x192; rw [hrip]
    have ht1fl : X86Int.cond t1.fl 4 = some true := by
      have hzf : t1.fl.zf = some true := by
        rw [ht1]
        show (aluF .and 0 0).2.zf = some true
        simp [aluF_zf, aluF_and_fst_self]
      have hcc := cond_cc4 hzf
      simpa using hcc
    have e2 : X86Int.step stage1Im t1 =
        .ok { t1 with rip := base + 0x1c7 } := by
      have h := step_jcc (im := stage1Im) (s := t1)
        (by rw [ht1rip]; exact hc.p192) ht1fl
      simpa using h
    obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
        { t1 with rip := base + 0x1c7 } := ⟨_, rfl⟩
    have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
      rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
    have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
      rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
    have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
      rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
    have ht2k : t2.k = s.k := by
      rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).k = s.k
      rw [ht1]
    refine ⟨2, t2, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ht2k⟩
    · rw [ht2]
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg 1 = some 0
      rw [reg_with_rip, ht1]
      show s.reg 1 = some 0
      exact hr1'
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg 6 = some c
      rw [reg_with_rip, ht1]
      show s.reg 6 = some c
      exact hr6
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg 15 = some e
      rw [reg_with_rip, ht1]
      show s.reg 15 = some e
      exact hr15
    · intro i _ _ _ _
      rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).reg i = s.reg i
      rw [reg_with_rip, ht1]
      rfl
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).mem = s.mem
      rw [mem_with_rip, ht1]
    · rw [ht2]
      show ({ t1 with rip := base + 0x1c7 }).regs.length = 16
      rw [len_with_rip, ht1]
      show s.regs.length = 16
      exact hlen16
  | cons d ds ih =>
    intro c' s hseq hrip hr1 hr6 hr15 hc'e heW hlenW hlen16
    cases hseq with
    | cons hd hrest =>
      have hle : c + 4 + d ≤ e := Nat.le_trans (DataSeq.le hrest) hc'e
      have hlencons : (d :: ds).length = ds.length + 1 :=
        List.length_cons
      obtain ⟨s1, hrun1, hs1rip, hs1r1, hs1r6, hs1r15, hs1fr, hs1mem,
          hs1len, hs1k⟩ :=
        parser_data_step_ok hrip hr1 hr6 hr15
          (by rw [hlencons]; omega) hlenW
          (by omega) hd hle heW hlen16
      have hs1r1' : s1.reg 1 = some ds.length := by
        rw [hlencons] at hs1r1
        simpa using hs1r1
      have hseq1 : DataSeq s1.mem (c + 4 + d) ds c' := by
        rw [hs1mem]; exact hrest
      obtain ⟨n2, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2r15, hs2fr,
          hs2mem, hs2len, hs2k⟩ :=
        ih hseq1 hs1rip hs1r1' hs1r6 hs1r15 hc'e heW
          (by rw [hlencons] at hlenW; omega) hs1len
      have hrun : run stage1Im (16 + n2) (Out.ok s) = Out.ok s2 := by
        rw [run_add stage1Im 16 n2, hrun1, hrun2]
      exact ⟨16 + n2, s2, hrun, hs2rip, hs2r1, hs2r6, hs2r15,
        fun i hi0 hi1 hi2 hi6 =>
          (hs2fr i hi0 hi1 hi2 hi6).trans (hs1fr i hi0 hi1 hi2 hi6),
        hs2mem.trans hs1mem, hs2len, hs2k.trans hs1k⟩

/-- The `nwords * 20` guard, success case: from `data_done`
(`base + 0x1c7`, `r10 = nwords`, cursor `rsi`, end `r15`) to
`base + 0x1ec`, when twenty bytes per word fit in what remains.
Only `rax` and `rdx` change. -/
theorem parser_guard_ok {c e nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1c7) (hr10 : s.reg 10 = some nw)
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hnw32 : nw < 2 ^ 32) (hce : c ≤ e) (heW : e < W)
    (hguard : 20 * nw ≤ e - c) (hlen16 : s.regs.length = 16) :
    ∃ s', run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x1ec ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have h2 : 2 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h4 : 4 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h8 : 8 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h16 : 16 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  have h20 : 20 * nw < W := by
    have hW : W = 2 ^ 64 := rfl; omega
  -- 0x1c7: mov rax, r10
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some nw) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p1c7) hr10
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some nw) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x1ca := by
    rw [ht1]; show s.rip + 3 = base + 0x1ca; rw [hrip]
  have ht1r0 : t1.reg 0 = some nw := by
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).reg 0 = some nw
    rw [reg_with_rip, reg_set_self (hlen 0 (by decide))]
  have ht1fr : ∀ i, i ≠ 0 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 0 (some nw) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  -- 0x1ca: add rax, rax (2*nw)
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 0 (some (aluF .add nw nw).1) with
        fl := (aluF .add nw nw).2, rip := t1.rip + 3 } :=
    step_alu_add (by rw [ht1rip]; exact hc.p1ca) ht1r0 ht1r0
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 0 (some (aluF .add nw nw).1) with
        fl := (aluF .add nw nw).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x1cd := by
    rw [ht2]; show t1.rip + 3 = base + 0x1cd; rw [ht1rip]
  have ht2r0 : t2.reg 0 = some (2 * nw) := by
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).reg 0 = some (2 * nw)
    rw [reg_set_self (by rw [ht1len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : nw + nw < W)]
    congr 1; omega
  have ht2fr : ∀ i, i ≠ 0 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 0 (some (aluF .add nw nw).1)).regs.length = 16
    rw [len_set]; exact ht1len
  -- 0x1cd: add rax, rax (4*nw)
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1) with
        fl := (aluF .add (2 * nw) (2 * nw)).2, rip := t2.rip + 3 } :=
    step_alu_add (by rw [ht2rip]; exact hc.p1cd) ht2r0 ht2r0
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1) with
        fl := (aluF .add (2 * nw) (2 * nw)).2, rip := t2.rip + 3 } :=
    ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x1d0 := by
    rw [ht3]; show t2.rip + 3 = base + 0x1d0; rw [ht2rip]
  have ht3r0 : t3.reg 0 = some (4 * nw) := by
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).reg 0 =
      some (4 * nw)
    rw [reg_set_self (by rw [ht2len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 2 * nw + 2 * nw < W)]
    congr 1; omega
  have ht3fr : ∀ i, i ≠ 0 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi]; exact ht2fr i hi
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show (t2.set 0 (some (aluF .add (2 * nw) (2 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht2len
  -- 0x1d0: mov rdx, rax (4*nw saved)
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 } :=
    step_mov (by rw [ht3rip]; exact hc.p1d0) ht3r0
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x1d3 := by
    rw [ht4]; show t3.rip + 3 = base + 0x1d3; rw [ht3rip]
  have ht4r2 : t4.reg 2 = some (4 * nw) := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).reg 2 =
      some (4 * nw)
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide)]
  have ht4r0 : t4.reg 0 = some (4 * nw) := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).reg 0 =
      some (4 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht3r0
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht3fr i hi0
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with rip := t3.rip + 3 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 2 (some (4 * nw)) with
      rip := t3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  -- 0x1d3: add rax, rax (8*nw)
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1) with
        fl := (aluF .add (4 * nw) (4 * nw)).2, rip := t4.rip + 3 } :=
    step_alu_add (by rw [ht4rip]; exact hc.p1d3) ht4r0 ht4r0
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1) with
        fl := (aluF .add (4 * nw) (4 * nw)).2, rip := t4.rip + 3 } :=
    ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x1d6 := by
    rw [ht5]; show t4.rip + 3 = base + 0x1d6; rw [ht4rip]
  have ht5r0 : t5.reg 0 = some (8 * nw) := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).reg 0 =
      some (8 * nw)
    rw [reg_set_self (by rw [ht4len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 4 * nw + 4 * nw < W)]
    congr 1; omega
  have ht5fr : ∀ i, i ≠ 0 → i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi0]; exact ht4fr i hi0 hi2
  have ht5r2 : t5.reg 2 = some (4 * nw) := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).reg 2 =
      some (4 * nw)
    rw [reg_set_other (by decide)]; exact ht4r2
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add (4 * nw) (4 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht4len
  -- 0x1d6: add rax, rax (16*nw)
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1) with
        fl := (aluF .add (8 * nw) (8 * nw)).2, rip := t5.rip + 3 } :=
    step_alu_add (by rw [ht5rip]; exact hc.p1d6) ht5r0 ht5r0
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1) with
        fl := (aluF .add (8 * nw) (8 * nw)).2, rip := t5.rip + 3 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x1d9 := by
    rw [ht6]; show t5.rip + 3 = base + 0x1d9; rw [ht5rip]
  have ht6r0 : t6.reg 0 = some (16 * nw) := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).reg 0 =
      some (16 * nw)
    rw [reg_set_self (by rw [ht5len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 8 * nw + 8 * nw < W)]
    congr 1; omega
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 2 → t6.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi0]; exact ht5fr i hi0 hi2
  have ht6r2 : t6.reg 2 = some (4 * nw) := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).reg 2 =
      some (4 * nw)
    rw [reg_set_other (by decide)]; exact ht5r2
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (8 * nw) (8 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht5len
  -- 0x1d9: add rax, rdx (20*nw)
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1) with
        fl := (aluF .add (16 * nw) (4 * nw)).2, rip := t6.rip + 3 } :=
    step_alu_add (by rw [ht6rip]; exact hc.p1d9) ht6r0 ht6r2
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1) with
        fl := (aluF .add (16 * nw) (4 * nw)).2, rip := t6.rip + 3 } :=
    ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x1dc := by
    rw [ht7]; show t6.rip + 3 = base + 0x1dc; rw [ht6rip]
  have ht7r0 : t7.reg 0 = some (20 * nw) := by
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).reg 0 =
      some (20 * nw)
    rw [reg_set_self (by rw [ht6len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega : 16 * nw + 4 * nw < W)]
    congr 1; omega
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → t7.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).reg i =
      s.reg i
    rw [reg_set_other hi0]; exact ht6fr i hi0 hi2
  have ht7r15 : t7.reg 15 = some e := by
    rw [ht7fr 15 (by decide) (by decide)]; exact hr15
  have ht7r6 : t7.reg 6 = some c := by
    rw [ht7fr 6 (by decide) (by decide)]; exact hr6
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).mem = s.mem
    rw [mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 0 (some (aluF .add (16 * nw) (4 * nw)).1)).regs.length =
      16
    rw [len_set]; exact ht6len
  -- 0x1dc: mov rdx, r15
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 2 (some e) with rip := t7.rip + 3 } :=
    step_mov (by rw [ht7rip]; exact hc.p1dc) ht7r15
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 2 (some e) with rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x1df := by
    rw [ht8]; show t7.rip + 3 = base + 0x1df; rw [ht7rip]
  have ht8r2 : t8.reg 2 = some e := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht7len]; decide)]
  have ht8r6 : t8.reg 6 = some c := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht7r6
  have ht8r0 : t8.reg 0 = some (20 * nw) := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg 0 =
      some (20 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht7r0
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → t8.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht7fr i hi0 hi2
  have ht8mem : t8.mem = s.mem := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht7len
  -- 0x1df: sub rdx, rsi (remaining)
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t8.rip + 3 } :=
    step_alu_sub (by rw [ht8rip]; exact hc.p1df) ht8r2 ht8r6
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t8.rip + 3 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x1e2 := by
    rw [ht9]; show t8.rip + 3 = base + 0x1e2; rw [ht8rip]
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht9r2 : t9.reg 2 = some (e - c) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht8len]; decide), hsub]
  have ht9r0 : t9.reg 0 = some (20 * nw) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).reg 0 = some (20 * nw)
    rw [reg_set_other (by decide)]; exact ht8r0
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → t9.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht8fr i hi0 hi2
  have ht9mem : t9.mem = s.mem := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht8len
  -- 0x1e2: cmp rax, rdx
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9 with fl := (aluF .sub (20 * nw) (e - c)).2, rip := t9.rip + 3 } :=
    step_cmp (by rw [ht9rip]; exact hc.p1e2) ht9r0 ht9r2
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9 with fl := (aluF .sub (20 * nw) (e - c)).2, rip := t9.rip + 3 } :=
    ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x1e5 := by
    rw [ht10]; show t9.rip + 3 = base + 0x1e5; rw [ht9rip]
  have ht10fl : X86Int.cond t10.fl 6 = some true := by
    by_cases hlt : 20 * nw < e - c
    · have hcf : t10.fl.cf = some true := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.cf = some true
        exact aluF_sub_cf_true hlt
      have hzf : t10.fl.zf =
          some ((aluF .sub (20 * nw) (e - c)).1 == 0) := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.zf =
          some ((aluF .sub (20 * nw) (e - c)).1 == 0)
        exact aluF_zf .sub _ _
      simp [X86Int.cond, hcf, hzf]
    · have heq : 20 * nw = e - c := by omega
      have hzval : (aluF .sub (20 * nw) (e - c)).1 = 0 := by
        rw [aluF_sub_fst,
          show 20 * nw + W - (e - c) = W by omega, Nat.mod_self]
      have hzf : t10.fl.zf = some true := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.zf = some true
        simp [aluF_zf, hzval]
      have hcf : t10.fl.cf = some false := by
        rw [ht10]
        show (aluF .sub (20 * nw) (e - c)).2.cf = some false
        exact aluF_sub_cf hlt
      simp [X86Int.cond, hcf, hzf]
  -- 0x1e5: jbe 0x1ec
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x1ec } := by
    have h := step_jcc (im := stage1Im) (s := t10)
      (by rw [ht10rip]; exact hc.p1e5) ht10fl
    simpa using h
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x1ec } := ⟨_, rfl⟩
  -- compose (11 steps)
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have hrun : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
    rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
  have ht11k : t11.k = s.k := by
    have h10k : t10.k = s.k := by
      have h9k : t9.k = s.k := by
        have h8k : t8.k = s.k := by
          have h7k : t7.k = s.k := by
            have h6k : t6.k = s.k := by
              have h5k : t5.k = s.k := by
                have h4k : t4.k = s.k := by
                  have h3k : t3.k = s.k := by
                    have h2k : t2.k = s.k := by
                      rw [ht2]
                      show (t1.set 0 (some (aluF .add nw nw).1)).k = s.k
                      rw [ht1]
                      show ({ s.set 0 (some nw) with
                        rip := s.rip + 3 }).k = s.k
                      rfl
                    rw [ht3]
                    show (t2.set 0
                      (some (aluF .add (2 * nw) (2 * nw)).1)).k = s.k
                    exact h2k
                  rw [ht4]
                  show ({ t3.set 2 (some (4 * nw)) with
                    rip := t3.rip + 3 }).k = s.k
                  exact h3k
                rw [ht5]
                show (t4.set 0
                  (some (aluF .add (4 * nw) (4 * nw)).1)).k = s.k
                exact h4k
              rw [ht6]
              show (t5.set 0
                (some (aluF .add (8 * nw) (8 * nw)).1)).k = s.k
              exact h5k
            rw [ht7]
            show (t6.set 0
              (some (aluF .add (16 * nw) (4 * nw)).1)).k = s.k
            exact h6k
          rw [ht8]
          show ({ t7.set 2 (some e) with rip := t7.rip + 3 }).k = s.k
          exact h7k
        rw [ht9]
        show (t8.set 2 (some (aluF .sub e c).1)).k = s.k
        exact h8k
      rw [ht10]
      show t9.k = s.k
      exact h9k
    rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).k = s.k
    exact h10k
  refine ⟨t11, hrun, ?_, ?_, ?_, ?_, ht11k⟩
  · rw [ht11]
  · intro i hi0 hi2
    have h10i : t10.reg i = s.reg i := by
      rw [ht10]; show t9.reg i = s.reg i; exact ht9fr i hi0 hi2
    rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).reg i = s.reg i
    rw [reg_with_rip]; exact h10i
  · rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).mem = s.mem
    rw [mem_with_rip, ht10]
    show t9.mem = s.mem
    exact ht9mem
  · rw [ht11]
    show ({ t10 with rip := base + 0x1ec }).regs.length = 16
    rw [len_with_rip, ht10]
    show t9.regs.length = 16
    exact ht9len

/-- The words-table `mmap`, success case: from `base + 0x1ec`
(cursor in `rsi`, `nwords` in `r10`) to `base + 0x22d`, saving
the cursor on the stack, asking the oracle for `64 * nwords`
bytes at `T`, and leaving the table base in `r14` with `rsp`
decremented by the saved cursor slot. -/
theorem parser_mmap1_ok {c nw sp T : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x1ec) (hr6 : s.reg 6 = some c)
    (hr10 : s.reg 10 = some nw) (hr4 : s.reg 4 = some sp)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hnw32 : nw < 2 ^ 32) (hnw0 : nw ≠ 0)
    (hhead : s.k.oracle.head? = some (.at T))
    (hTpage : T % page = 0) (hTpos : 0 < T)
    (hTbound : T + X86Env.pages (64 * nw) ≤ 2 ^ 47)
    (hfree : s.mem.free T (X86Env.pages (64 * nw)) = true)
    (hlen16 : s.regs.length = 16) :
    ∃ (s' : X86Int.St) (m2 : Mem), run stage1Im 20 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x22d ∧ s'.reg 14 = some T ∧
      s'.reg 4 = some (sp - 8) ∧
      s'.mem = m2 ++
        [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] ∧
      (∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x → m2.rdN x k = s.mem.rdN x k) ∧
      rd64 s'.mem (sp - 8) = some (c % W) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → i ≠ 14 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = (Kern.next s.k).2 ∧
      wshape m2 = wshape s.mem := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hsp16mod : (sp + W - 16) % W = sp - 16 := by
    have h : sp + W - 16 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have haddr2 : (sp - 8 + W - 8) % W = sp - 16 := by
    have h : sp - 8 + W - 8 = (sp - 16) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have h64 : ∀ k : Nat, k ≤ 64 → k * nw < W := by
    intro k hk
    have hW : W = 2 ^ 64 := rfl
    have h1 : k * nw ≤ 64 * nw := Nat.mul_le_mul_right nw hk
    have h2a : nw ≤ 2 ^ 32 - 1 := by omega
    have h2b : 64 * nw ≤ 64 * (2 ^ 32 - 1) :=
      Nat.mul_le_mul_left 64 h2a
    have h2c : 64 * (2 ^ 32 - 1) < 2 ^ 64 := by decide
    omega
  -- stack writes, obtained once
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W) c =
      some m :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) c)
  have hm1 : wr64 s.mem ((sp + W - 8) % W) c = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) c = some m1 := by
    rwa [hsp8mod] at hm1
  have hw1 : wshape m1 = wshape s.mem := wrN_wshape _ _ _ hm1raw
  -- 0x1ec: push rsi
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 1 } :=
    step_push (by rw [hrip]; exact hc.p1ec) hr6 hr4 hm1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with mem := m1, rip := s.rip + 1 } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x1ed := by
    rw [ht1]; show s.rip + 1 = base + 0x1ed; rw [hrip]
  have ht1r4 : t1.reg 4 = some (sp - 8) := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (hlen 4 (by decide)), hsp8mod]
  have ht1r10 : t1.reg 10 = some nw := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 10 = some nw
    rw [reg_set_other (by decide)]; exact hr10
  have ht1fr : ∀ i, i ≠ 4 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have ht1mem : t1.mem = m1 := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have ht1k : t1.k = s.k := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).k = s.k
    rfl
  -- 0x1ed: mov32 rdx, r10d
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 } :=
    step_mov32 (by rw [ht1rip]; exact hc.p1ed) ht1r10
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x1f0 := by
    rw [ht2]; show t1.rip + 3 = base + 0x1f0; rw [ht1rip]
  have hnwmod : nw % 2 ^ 32 = nw := Nat.mod_eq_of_lt hnw32
  have ht2r2 : t2.reg 2 = some nw := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).reg 2 =
      some nw
    rw [reg_with_rip, reg_set_self (by rw [ht1len]; decide), hnwmod]
  have ht2fr : ∀ i, i ≠ 2 → i ≠ 4 → t2.reg i = s.reg i := by
    intro i hi2 hi4
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht1fr i hi4
  have ht2r4 : t2.reg 4 = some (sp - 8) := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht1r4
  have ht2mem : t2.mem = m1 := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with
      rip := t1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht1len
  have ht2k : t2.k = s.k := by
    rw [ht2]
    show ({ t1.set 2 (some (nw % 2 ^ 32)) with rip := t1.rip + 3 }).k = s.k
    exact ht1k
  -- six doublings of rdx: nw -> 64*nw, collected as one run of 6
  have hdbl : ∀ (u : X86Int.St) (site : Nat) (v : Nat),
      stage1Im site = some (.alu .add 2 2, 3) → u.rip = site →
      u.reg 2 = some v → v + v < W → u.regs.length = 16 →
      ∃ u', run stage1Im 1 (Out.ok u) = Out.ok u' ∧
        u'.rip = site + 3 ∧ u'.reg 2 = some (2 * v) ∧
        (∀ i, i ≠ 2 → u'.reg i = u.reg i) ∧ u'.mem = u.mem ∧
        u'.regs.length = 16 ∧ u'.k = u.k := by
    intro u site v hsite hurip hur2 hvW hulen
    have e : X86Int.step stage1Im u =
        .ok { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } :=
      step_alu_add (by rw [hurip]; exact hsite) hur2 hur2
    obtain ⟨u', hu'⟩ : ∃ t : X86Int.St, t =
        { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } := ⟨_, rfl⟩
    have hrun : run stage1Im 1 (Out.ok u) = Out.ok u' := by
      rw [run_succ_ok stage1Im 0 u, e, ← hu', run_zero]
    refine ⟨u', hrun, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu']; show u.rip + 3 = site + 3; rw [hurip]
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).reg 2 = some (2 * v)
      rw [reg_set_self (by rw [hulen]; decide), aluF_add_fst,
        Nat.mod_eq_of_lt hvW, Nat.two_mul]
    · intro i hi
      rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).reg i = u.reg i
      rw [reg_set_other hi]
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).mem = u.mem
      rw [mem_set]
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).regs.length = 16
      rw [len_set]; exact hulen
    · rw [hu']
      show (u.set 2 (some (aluF .add v v).1)).k = u.k
      rfl
  obtain ⟨t3, hrun3, ht3rip, ht3r2, ht3fr, ht3mem, ht3len, ht3k⟩ :=
    hdbl t2 (base + 0x1f0) nw hc.p1f0 ht2rip ht2r2
      (by have := h64 2 (by decide); omega) ht2len
  obtain ⟨t4, hrun4, ht4rip, ht4r2, ht4fr, ht4mem, ht4len, ht4k⟩ :=
    hdbl t3 (base + 0x1f3) (2 * nw) (by rw [show base + 0x1f0 + 3 = base + 0x1f3 from rfl] at ht3rip; exact hc.p1f3)
      (by rw [ht3rip]) ht3r2
      (by have := h64 4 (by decide); omega) ht3len
  obtain ⟨t5, hrun5, ht5rip, ht5r2, ht5fr, ht5mem, ht5len, ht5k⟩ :=
    hdbl t4 (base + 0x1f6) (2 * (2 * nw)) (by exact hc.p1f6)
      (by rw [ht4rip]) ht4r2
      (by have := h64 8 (by decide); omega) ht4len
  obtain ⟨t6, hrun6, ht6rip, ht6r2, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    hdbl t5 (base + 0x1f9) (2 * (2 * (2 * nw))) (by exact hc.p1f9)
      (by rw [ht5rip]) ht5r2
      (by have := h64 16 (by decide); omega) ht5len
  obtain ⟨t7, hrun7, ht7rip, ht7r2, ht7fr, ht7mem, ht7len, ht7k⟩ :=
    hdbl t6 (base + 0x1fc) (2 * (2 * (2 * (2 * nw)))) (by exact hc.p1fc)
      (by rw [ht6rip]) ht6r2
      (by have := h64 32 (by decide); omega) ht6len
  obtain ⟨t8, hrun8, ht8rip, ht8r2, ht8fr, ht8mem, ht8len, ht8k⟩ :=
    hdbl t7 (base + 0x1ff) (2 * (2 * (2 * (2 * (2 * nw)))))
      (by exact hc.p1ff) (by rw [ht7rip]) ht7r2
      (by have := h64 64 (by decide); omega) ht7len
  have ht8rip' : t8.rip = base + 0x202 := ht8rip
  have ht8r2' : t8.reg 2 = some (64 * nw) := by
    have h : 2 * (2 * (2 * (2 * (2 * (2 * nw))))) = 64 * nw := by omega
    rw [h] at ht8r2; exact ht8r2
  have ht8r4 : t8.reg 4 = some (sp - 8) := by
    rw [ht8fr 4 (by decide), ht7fr 4 (by decide), ht6fr 4 (by decide),
      ht5fr 4 (by decide), ht4fr 4 (by decide), ht3fr 4 (by decide)]
    exact ht2r4
  have ht8frs : ∀ i, i ≠ 2 → i ≠ 4 → t8.reg i = s.reg i := by
    intro i hi2 hi4
    rw [ht8fr i hi2, ht7fr i hi2, ht6fr i hi2, ht5fr i hi2, ht4fr i hi2,
      ht3fr i hi2]
    exact ht2fr i hi2 hi4
  have ht8mems : t8.mem = m1 := by
    rw [ht8mem, ht7mem, ht6mem, ht5mem, ht4mem, ht3mem]; exact ht2mem
  have ht8ks : t8.k = s.k := by
    rw [ht8k, ht7k, ht6k, ht5k, ht4k, ht3k]; exact ht2k
  -- 0x202: movI rax, 9
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 0 (some (9 % W)) with rip := t8.rip + 5 } :=
    step_movI (by rw [ht8rip']; exact hc.p202)
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 0 (some (9 % W)) with rip := t8.rip + 5 } := ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x207 := by
    rw [ht9]; show t8.rip + 5 = base + 0x207; rw [ht8rip']
  have h9W : 9 % W = 9 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht9r0 : t9.reg 0 = some 9 := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_self (by rw [ht8len]; decide), h9W]
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → t9.reg i = s.reg i := by
    intro i hi0 hi2 hi4
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht8frs i hi2 hi4
  have ht9r2 : t9.reg 2 = some (64 * nw) := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg 2 =
      some (64 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht8r2'
  have ht9r4 : t9.reg 4 = some (sp - 8) := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht8r4
  have ht9r7 : ∃ v, t9.reg 7 = some v := by
    obtain ⟨v, hv⟩ := hr7
    exact ⟨v, by rw [ht9fr 7 (by decide) (by decide) (by decide)]; exact hv⟩
  have ht9r9 : ∃ v, t9.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by rw [ht9fr 9 (by decide) (by decide) (by decide)]; exact hv⟩
  have ht9mem : t9.mem = m1 := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht8mems
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht8len
  have ht9k : t9.k = s.k := by
    rw [ht9]
    show ({ t8.set 0 (some (9 % W)) with rip := t8.rip + 5 }).k = s.k
    exact ht8ks
  -- 0x207: xor rdi, rdi
  obtain ⟨v7, hv7⟩ := ht9r7
  have hxor : ∀ a : Nat, (aluF .xor a a).1 = 0 := by
    intro a; simp [aluF, Nat.xor_self]
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t9.rip + 2 } :=
    step_alu_xor (by rw [ht9rip]; exact hc.p207) hv7 hv7
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t9.rip + 2 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x209 := by
    rw [ht10]; show t9.rip + 2 = base + 0x209; rw [ht9rip]
  have ht10r7 : t10.reg 7 = some 0 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 7 = some 0
    rw [reg_set_self (by rw [ht9len]; decide), hxor]
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 7 → t10.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi7
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg i = s.reg i
    rw [reg_set_other hi7]; exact ht9fr i hi0 hi2 hi4
  have ht10r0 : t10.reg 0 = some 9 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 0 = some 9
    rw [reg_set_other (by decide)]; exact ht9r0
  have ht10r2 : t10.reg 2 = some (64 * nw) := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 2 = some (64 * nw)
    rw [reg_set_other (by decide)]; exact ht9r2
  have ht10r4 : t10.reg 4 = some (sp - 8) := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide)]; exact ht9r4
  have ht10mem : t10.mem = m1 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).mem = m1
    rw [mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).regs.length = 16
    rw [len_set]; exact ht9len
  have ht10k : t10.k = s.k := by
    rw [ht10]
    show (t9.set 7 (some (aluF .xor v7 v7).1)).k = s.k
    exact ht9k
  -- 0x209: mov rsi, rdx
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 } :=
    step_mov (by rw [ht10rip]; exact hc.p209) ht10r2
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x20c := by
    rw [ht11]; show t10.rip + 3 = base + 0x20c; rw [ht10rip]
  have ht11r6 : t11.reg 6 = some (64 * nw) := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 6 =
      some (64 * nw)
    rw [reg_with_rip, reg_set_self (by rw [ht10len]; decide)]
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t11.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi6]; exact ht10fr i hi0 hi2 hi4 hi7
  have ht11r0 : t11.reg 0 = some 9 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r0
  have ht11r7 : t11.reg 7 = some 0 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 7 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r7
  have ht11r4 : t11.reg 4 = some (sp - 8) := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r4
  have ht11mem : t11.mem = m1 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht10mem
  have ht11len : t11.regs.length = 16 := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with
      rip := t10.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht10len
  have ht11k : t11.k = s.k := by
    rw [ht11]
    show ({ t10.set 6 (some (64 * nw)) with rip := t10.rip + 3 }).k = s.k
    exact ht10k
  -- 0x20c: movI rdx, 3
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (3 % W)) with rip := t11.rip + 5 } :=
    step_movI (by rw [ht11rip]; exact hc.p20c)
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (3 % W)) with rip := t11.rip + 5 } := ⟨_, rfl⟩
  have ht12rip : t12.rip = base + 0x211 := by
    rw [ht12]; show t11.rip + 5 = base + 0x211; rw [ht11rip]
  have h3W : 3 % W = 3 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht12r2 : t12.reg 2 = some 3 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 2 =
      some 3
    rw [reg_with_rip, reg_set_self (by rw [ht11len]; decide), h3W]
  have ht12fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t12.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht11fr i hi0 hi2 hi4 hi6 hi7
  have ht12r0 : t12.reg 0 = some 9 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 0 =
      some 9
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r0
  have ht12r6 : t12.reg 6 = some (64 * nw) := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 6 =
      some (64 * nw)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r6
  have ht12r7 : t12.reg 7 = some 0 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 7 =
      some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r7
  have ht12r4 : t12.reg 4 = some (sp - 8) := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r4
  have ht12mem : t12.mem = m1 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht11mem
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with
      rip := t11.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht11len
  have ht12k : t12.k = s.k := by
    rw [ht12]
    show ({ t11.set 2 (some (3 % W)) with rip := t11.rip + 5 }).k = s.k
    exact ht11k
  -- 0x211: movI r10, 34
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12.set 10 (some (34 % W)) with rip := t12.rip + 6 } :=
    step_movI (by rw [ht12rip]; exact hc.p211)
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12.set 10 (some (34 % W)) with rip := t12.rip + 6 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x217 := by
    rw [ht13]; show t12.rip + 6 = base + 0x217; rw [ht12rip]
  have h34W : 34 % W = 34 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht13r10 : t13.reg 10 = some 34 := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_self (by rw [ht12len]; decide), h34W]
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t13.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi10
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi10]
    exact ht12fr i hi0 hi2 hi4 hi6 hi7
  have ht13r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 →
      t13.reg j = t12.reg j := by
    intro j hj
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).reg j =
      t12.reg j
    rw [reg_with_rip, reg_set_other (by omega)]
  have ht13mem : t13.mem = m1 := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with
      rip := t12.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht12len
  have ht13k : t13.k = s.k := by
    rw [ht13]
    show ({ t12.set 10 (some (34 % W)) with rip := t12.rip + 6 }).k = s.k
    exact ht12k
  -- 0x217: pushI -1
  have ht13r4 : t13.reg 4 = some (sp - 8) := by
    rw [ht13r 4 (Or.inr (Or.inr (Or.inl rfl)))]; exact ht12r4
  obtain ⟨m2, hm2⟩ : ∃ m, wr64 m1 ((sp - 8 + W - 8) % W)
      (18446744073709551615 % W) = some m := by
    have hh : (wr64 s.mem (sp - 16) (18446744073709551615 % W)).isSome := by
      have h2 : (wr64 s.mem ((sp + W - 16) % W)
          (18446744073709551615 % W)).isSome :=
        hroom sp hr4 1 (by decide) _
      rwa [hsp16mod] at h2
    have h3 := wr64_some_of_wr64 hm1wr hh
    rw [haddr2]
    exact Option.isSome_iff_exists.mp h3
  have hm2wr : wr64 m1 (sp - 16) (18446744073709551615 % W) = some m2 := by
    rwa [haddr2] at hm2
  have hw2 : wshape m2 = wshape s.mem :=
    (wrN_wshape _ _ _ hm2wr).trans hw1
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13.set 4 (some ((sp - 8 + W - 8) % W)) with mem := m2, rip := t13.rip + 2 } := by
    have hm2' : wr64 t13.mem ((sp - 8 + W - 8) % W)
        (18446744073709551615 % W) = some m2 := by
      rw [ht13mem]; exact hm2
    exact step_pushI (by rw [ht13rip]; exact hc.p217) ht13r4 hm2'
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13.set 4 (some ((sp - 8 + W - 8) % W)) with mem := m2, rip := t13.rip + 2 } :=
    ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x219 := by
    rw [ht14]; show t13.rip + 2 = base + 0x219; rw [ht13rip]
  have ht14r4 : t14.reg 4 = some (sp - 16) := by
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).reg 4 = some (sp - 16)
    rw [reg_set_self (by rw [ht13len]; decide), haddr2]
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t14.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi10
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]; exact ht13fr i hi0 hi2 hi4 hi6 hi7 hi10
  have ht14r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t14.reg j = t13.reg j := by
    intro j hj
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).reg j = t13.reg j
    rw [reg_set_other (by omega)]
  have ht14mem : t14.mem = m2 := by rw [ht14]
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).regs.length = 16
    rw [len_set]; exact ht13len
  have ht14k : t14.k = s.k := by
    rw [ht14]
    show (t13.set 4 (some ((sp - 8 + W - 8) % W))).k = s.k
    exact ht13k
  -- 0x219: pop r8
  have hImm : 18446744073709551615 % W = W - 1 := by
    have h : (18446744073709551615 : Nat) = W - 1 := by
      unfold W; decide
    rw [h]
    exact Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)
  have hrd64m2 : rd64 m2 (sp - 16) = some (W - 1) := by
    have h := rd64_wr64 hm2wr
    rwa [hImm, Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)] at h
  have e15 : X86Int.step stage1Im t14 =
      .ok { (t14.set 4 (some ((sp - 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := t14.rip + 2 } := by
    have hv : rd64 t14.mem (sp - 16) = some (W - 1) := by
      rw [ht14mem]; exact hrd64m2
    exact step_pop (by rw [ht14rip]; exact hc.p219) ht14r4 hv
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { (t14.set 4 (some ((sp - 16 + 8) % W))).set 8 (some (W - 1)) with
        rip := t14.rip + 2 } := ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x21b := by
    rw [ht15]; show t14.rip + 2 = base + 0x21b; rw [ht14rip]
  have hspret : (sp - 16 + 8) % W = sp - 8 := by
    rw [show sp - 16 + 8 = sp - 8 by omega,
      Nat.mod_eq_of_lt (by omega)]
  have ht15r8 : t15.reg 8 = some (W - 1) := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg 8 = some (W - 1)
    rw [reg_set_self (by
      show 8 < (t14.set 4 (some ((sp - 16 + 8) % W))).regs.length
      rw [len_set, ht14len]; decide)]
  have ht15r4 : t15.reg 4 = some (sp - 8) := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide),
      reg_set_self (by rw [ht14len]; decide), hspret]
  have ht15fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 10 → t15.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi10
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg i = s.reg i
    rw [reg_set_other hi8, reg_set_other hi4]
    exact ht14fr i hi0 hi2 hi4 hi6 hi7 hi10
  have ht15r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t15.reg j = t14.reg j := by
    intro j hj
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).reg j = t14.reg j
    rw [reg_set_other (by omega), reg_set_other (by omega)]
  have ht15mem : t15.mem = m2 := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).mem = m2
    rw [mem_set, mem_set]; exact ht14mem
  have ht15len : t15.regs.length = 16 := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).regs.length = 16
    rw [len_set, len_set]; exact ht14len
  have ht15k : t15.k = s.k := by
    rw [ht15]
    show ((t14.set 4 (some ((sp - 16 + 8) % W))).set 8
      (some (W - 1))).k = s.k
    exact ht14k
  -- 0x21b: xor r9, r9
  have ht15r9v : ∃ v, t15.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by
      rw [ht15fr 9 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)]
      exact hv⟩
  obtain ⟨v9, hv9⟩ := ht15r9v
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t15.rip + 3 } :=
    step_alu_xor (by rw [ht15rip]; exact hc.p21b) hv9 hv9
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t15.rip + 3 } := ⟨_, rfl⟩
  have ht16rip : t16.rip = base + 0x21e := by
    rw [ht16]; show t15.rip + 3 = base + 0x21e; rw [ht15rip]
  have ht16r9 : t16.reg 9 = some 0 := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).reg 9 = some 0
    rw [reg_set_self (by rw [ht15len]; decide), hxor]
  have ht16fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 9 → i ≠ 10 → t16.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).reg i = s.reg i
    rw [reg_set_other hi9]
    exact ht15fr i hi0 hi2 hi4 hi6 hi7 hi8 hi10
  have ht16r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 ∨
      j = 8 ∨ j = 10 → t16.reg j = t15.reg j := by
    intro j hj
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).reg j = t15.reg j
    rw [reg_set_other (by omega)]
  have ht16mem : t16.mem = m2 := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).mem = m2
    rw [mem_set]; exact ht15mem
  have ht16len : t16.regs.length = 16 := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).regs.length = 16
    rw [len_set]; exact ht15len
  have ht16k : t16.k = s.k := by
    rw [ht16]
    show (t15.set 9 (some (aluF .xor v9 v9).1)).k = s.k
    exact ht15k
  have ht16r0 : t16.reg 0 = some 9 := by
    rw [ht16r 0 (Or.inl rfl), ht15r 0 (Or.inl rfl),
      ht14r 0 (Or.inl rfl), ht13r 0 (Or.inl rfl)]
    exact ht12r0
  have ht16r7 : t16.reg 7 = some 0 := by
    rw [ht16r 7 (by decide), ht15r 7 (by decide), ht14r 7 (by decide),
      ht13r 7 (by decide)]
    exact ht12r7
  have ht16r6 : t16.reg 6 = some (64 * nw) := by
    rw [ht16r 6 (by decide), ht15r 6 (by decide), ht14r 6 (by decide),
      ht13r 6 (by decide)]
    exact ht12r6
  have ht16r2 : t16.reg 2 = some 3 := by
    rw [ht16r 2 (by decide), ht15r 2 (by decide), ht14r 2 (by decide),
      ht13r 2 (by decide)]
    exact ht12r2
  have ht16r10 : t16.reg 10 = some 34 := by
    rw [ht16r 10 (by decide), ht15r 10 (by decide), ht14r 10 (by decide)]
    exact ht13r10
  have ht16r8 : t16.reg 8 = some (W - 1) := by
    rw [ht16r 8 (by decide)]
    exact ht15r8
  -- 0x21e: syscall (mmap)
  have hm2free : m2.free T (X86Env.pages (64 * nw)) = true := by
    rw [free_of_wshape hw2]; exact hfree
  have hplace : X86Env.place m2 (some (.at T)) (X86Env.pages (64 * nw)) =
      some (some T) :=
    place_at hTpage hTpos hTbound hm2free
  have hsc : X86Env.syscall t16.k t16.mem 9 0 (64 * nw) PROT_RW
      MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret T (Kern.next s.k).2
        (m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩])) := by
    rw [ht16k, ht16mem]
    exact syscall_mmap_exact (by
      have : 0 < nw := Nat.pos_of_ne_zero hnw0; omega) hhead hplace
  have e17 : X86Int.step stage1Im t16 =
      .ok { ((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩], rip := t16.rip + 2 } :=
    step_syscall_ret (by rw [ht16rip]; exact hc.p21e) ht16r0 ht16r7
      ht16r6 ht16r2 ht16r10 ht16r8 ht16r9 hsc
  obtain ⟨t17, ht17⟩ : ∃ t : X86Int.St, t =
      { ((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩], rip := t16.rip + 2 } :=
    ⟨_, rfl⟩
  have ht17rip : t17.rip = base + 0x220 := by
    rw [ht17]; show t16.rip + 2 = base + 0x220; rw [ht16rip]
  have ht17r0 : t17.reg 0 = some T := by
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).reg 0 = some T
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_self (by rw [ht16len]; decide)]
  have ht17fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t17.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).reg i = s.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
    exact ht16fr i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
  have ht17r4 : t17.reg 4 = some (sp - 8) := by
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).reg 4 = some (sp - 8)
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_other (by decide),
      ht16r 4 (Or.inr (Or.inr (Or.inl rfl)))]
    exact ht15r4
  have ht17mem : t17.mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
    rw [ht17]
  have ht17len : t17.regs.length = 16 := by
    rw [ht17]
    show (((t16.set 0 (some T)).set 1 (some (t16.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact ht16len
  have ht17k : t17.k = (Kern.next s.k).2 := by
    rw [ht17]
  -- 0x220: test rax, rax
  have e18 : X86Int.step stage1Im t17 =
      .ok { t17 with fl := (aluF .and T T).2, rip := t17.rip + 3 } :=
    step_test (by rw [ht17rip]; exact hc.p220) ht17r0 ht17r0
  obtain ⟨t18, ht18⟩ : ∃ t : X86Int.St, t =
      { t17 with fl := (aluF .and T T).2, rip := t17.rip + 3 } := ⟨_, rfl⟩
  have ht18rip : t18.rip = base + 0x223 := by
    rw [ht18]; show t17.rip + 3 = base + 0x223; rw [ht17rip]
  have hmsb : msb T = false := by
    have hT63 : T < 2 ^ 63 := by
      have hpg : 0 ≤ X86Env.pages (64 * nw) := Nat.zero_le _
      have h47 : (2 : Nat) ^ 47 < 2 ^ 63 := by decide
      omega
    unfold msb
    rw [Nat.div_eq_of_lt hT63]
    rfl
  have ht18fl : X86Int.cond t18.fl 9 = some true := by
    have hsf : t18.fl.sf = some false := by
      rw [ht18]
      show (aluF .and T T).2.sf = some false
      have h1 : (aluF .and T T).2.sf = some (msb (aluF .and T T).1) := rfl
      rw [h1, aluF_and_fst_self, hmsb]
    simp [X86Int.cond, hsf]
  -- 0x223: jns 0x22a
  have e19 : X86Int.step stage1Im t18 =
      .ok { t18 with rip := base + 0x22a } := by
    have h := step_jcc (im := stage1Im) (s := t18)
      (by rw [ht18rip]; exact hc.p223) ht18fl
    simpa using h
  obtain ⟨t19, ht19⟩ : ∃ t : X86Int.St, t =
      { t18 with rip := base + 0x22a } := ⟨_, rfl⟩
  have ht19rip : t19.rip = base + 0x22a := by rw [ht19]
  have ht19r0 : t19.reg 0 = some T := by
    have h18r0 : t18.reg 0 = some T := by
      rw [ht18]; show t17.reg 0 = some T; exact ht17r0
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).reg 0 = some T
    rw [reg_with_rip]; exact h18r0
  have ht19fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t19.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h18i : t18.reg i = s.reg i := by
      rw [ht18]; show t17.reg i = s.reg i
      exact ht17fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).reg i = s.reg i
    rw [reg_with_rip]; exact h18i
  have ht19r4 : t19.reg 4 = some (sp - 8) := by
    have h18r4 : t18.reg 4 = some (sp - 8) := by
      rw [ht18]; show t17.reg 4 = some (sp - 8); exact ht17r4
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).reg 4 = some (sp - 8)
    rw [reg_with_rip]; exact h18r4
  have ht19mem : t19.mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
    have h18m : t18.mem =
        m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
      rw [ht18]; show t17.mem = _; exact ht17mem
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]
    rw [mem_with_rip]; exact h18m
  have ht19len : t19.regs.length = 16 := by
    have h18l : t18.regs.length = 16 := by
      rw [ht18]; show t17.regs.length = 16; exact ht17len
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).regs.length = 16
    rw [len_with_rip]; exact h18l
  have ht19k : t19.k = (Kern.next s.k).2 := by
    have h18k : t18.k = (Kern.next s.k).2 := by
      rw [ht18]; show t17.k = (Kern.next s.k).2; exact ht17k
    rw [ht19]
    show ({ t18 with rip := base + 0x22a }).k = (Kern.next s.k).2
    exact h18k
  -- 0x22a: mov r14, rax
  have e20 : X86Int.step stage1Im t19 =
      .ok { t19.set 14 (some T) with rip := t19.rip + 3 } :=
    step_mov (by rw [ht19rip]; exact hc.p22a) ht19r0
  obtain ⟨t20, ht20⟩ : ∃ t : X86Int.St, t =
      { t19.set 14 (some T) with rip := t19.rip + 3 } := ⟨_, rfl⟩
  -- compose the 20 steps
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  have r16 : run stage1Im 1 (Out.ok t15) = Out.ok t16 := by
    rw [run_succ_ok stage1Im 0 t15, e16, ← ht16, run_zero]
  have r17 : run stage1Im 1 (Out.ok t16) = Out.ok t17 := by
    rw [run_succ_ok stage1Im 0 t16, e17, ← ht17, run_zero]
  have r18 : run stage1Im 1 (Out.ok t17) = Out.ok t18 := by
    rw [run_succ_ok stage1Im 0 t17, e18, ← ht18, run_zero]
  have r19 : run stage1Im 1 (Out.ok t18) = Out.ok t19 := by
    rw [run_succ_ok stage1Im 0 t18, e19, ← ht19, run_zero]
  have r20 : run stage1Im 1 (Out.ok t19) = Out.ok t20 := by
    rw [run_succ_ok stage1Im 0 t19, e20, ← ht20, run_zero]
  have hrun : run stage1Im 20 (Out.ok s) = Out.ok t20 := by
    rw [show (20 : Nat) = 1 + 19 by omega, run_add stage1Im 1 19, r1,
      show (19 : Nat) = 1 + 18 by omega, run_add stage1Im 1 18, r2,
      show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, hrun3,
      show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, hrun4,
      show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, hrun5,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, hrun6,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, hrun7,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, hrun8,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r9,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r10,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r11,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r12,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r13,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r14,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r15,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r16,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r17,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r18,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r19, r20]
  have hm2fr : ∀ x k, x + k ≤ sp - 16 ∨ sp ≤ x →
      m2.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    have h1 : m2.rdN x k = m1.rdN x k := by
      apply rdN_wr64_other hm2wr
      rcases hdis with h | h
      · exact Or.inr h
      · exact Or.inl (by omega)
    rw [h1]
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr (by omega)
    · exact Or.inl (by omega)
  have hrd64cur : rd64 t20.mem (sp - 8) = some (c % W) := by
    have h20mem : t20.mem =
        m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩] := by
      rw [ht20]
      show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).mem =
        m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]
      rw [mem_with_rip, mem_set]; exact ht19mem
    have hm2cur : rd64 m2 (sp - 8) = some (c % W) := by
      have h1 : rd64 m2 (sp - 8) = rd64 m1 (sp - 8) :=
        rd64_wr64_other hm2wr (Or.inl (by omega))
      rw [h1]; exact rd64_wr64 hm1wr
    obtain ⟨bs8, hbs8, hle⟩ : ∃ bs, m2.rdN (sp - 8) 8 = some bs ∧
        leVal bs = c % W := by
      have h := hm2cur
      unfold rd64 at h
      have hs : (m2.rdN (sp - 8) 8).isSome := by
        cases h8 : m2.rdN (sp - 8) 8 with
        | none => rw [h8] at h; simp at h
        | some bs => simp
      obtain ⟨bs, hbs⟩ := Option.isSome_iff_exists.mp hs
      exact ⟨bs, hbs, by rw [hbs] at h; exact Option.some.inj h⟩
    rw [h20mem]
    show ((m2 ++ [(⟨T, List.replicate (X86Env.pages (64 * nw)) 0,
      true⟩ : Region)]).rdN (sp - 8) 8).map leVal = some (c % W)
    rw [rdN_append_left hbs8]
    simp [hle]
  refine ⟨t20, m2, hrun, ?_, ?_, ?_, ?_, hm2fr, hrd64cur, ?_, ?_, ?_,
    hw2⟩
  · rw [ht20]; show t19.rip + 3 = base + 0x22d; rw [ht19rip]
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).reg 14 =
      some T
    rw [reg_with_rip, reg_set_self (by rw [ht19len]; decide)]
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht19r4
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).mem =
      m2 ++ [⟨T, List.replicate (X86Env.pages (64 * nw)) 0, true⟩]
    rw [mem_with_rip, mem_set]; exact ht19mem
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11 hi14
    rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi14]
    exact ht19fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht19len
  · rw [ht20]
    show ({ t19.set 14 (some T) with rip := t19.rip + 3 }).k =
      (Kern.next s.k).2
    exact ht19k

/-- The scratch `mmap`, success case: from `base + 0x22d`
(`r13` still the file length, `rsp` decremented by the saved
cursor) to `base + 0x258`, asking the oracle for `L` bytes at
`S` and leaving the scratch base in `r11`. -/
theorem parser_mmap2_ok {L sp1 S : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x22d) (hr13 : s.reg 13 = some L)
    (hr4 : s.reg 4 = some sp1)
    (hr7 : ∃ v, s.reg 7 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hroom : StackRoom s 1) (hsp8 : 8 ≤ sp1) (hspW : sp1 < W)
    (hL : L ≠ 0)
    (hhead : s.k.oracle.head? = some (.at S))
    (hSpage : S % page = 0) (hSpos : 0 < S)
    (hSbound : S + X86Env.pages L ≤ 2 ^ 47)
    (hfree : s.mem.free S (X86Env.pages L) = true)
    (hlen16 : s.regs.length = 16) :
    ∃ (s' : X86Int.St) (m1 : Mem),
      run stage1Im 12 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x258 ∧ s'.reg 11 = some S ∧
      s'.reg 4 = some sp1 ∧
      s'.mem = m1 ++
        [⟨S, List.replicate (X86Env.pages L) 0, true⟩] ∧
      (∀ x k, x + k ≤ sp1 - 8 ∨ sp1 ≤ x →
        m1.rdN x k = s.mem.rdN x k) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
        i ≠ 9 → i ≠ 10 → i ≠ 11 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = (Kern.next s.k).2 ∧
      wshape m1 = wshape s.mem := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hsp8mod : (sp1 + W - 8) % W = sp1 - 8 := by
    have h : sp1 + W - 8 = (sp1 - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have hxor : ∀ a : Nat, (aluF .xor a a).1 = 0 := by
    intro a; simp [aluF, Nat.xor_self]
  -- 0x22d: movI rax, 9
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 0 (some (9 % W)) with rip := s.rip + 5 } :=
    step_movI (by rw [hrip]; exact hc.p22d)
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 0 (some (9 % W)) with rip := s.rip + 5 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x232 := by
    rw [ht1]; show s.rip + 5 = base + 0x232; rw [hrip]
  have h9W : 9 % W = 9 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht1r0 : t1.reg 0 = some 9 := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_self (hlen 0 (by decide)), h9W]
  have ht1fr : ∀ i, i ≠ 0 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1r4 : t1.reg 4 = some sp1 := by
    rw [ht1fr 4 (by decide)]; exact hr4
  have ht1r13 : t1.reg 13 = some L := by
    rw [ht1fr 13 (by decide)]; exact hr13
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by
    rw [ht1]
    show ({ s.set 0 (some (9 % W)) with rip := s.rip + 5 }).k = s.k
    rfl
  -- 0x232: xor rdi, rdi
  obtain ⟨v7, hv7⟩ : ∃ v, t1.reg 7 = some v := by
    obtain ⟨v, hv⟩ := hr7
    exact ⟨v, by rw [ht1fr 7 (by decide)]; exact hv⟩
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t1.rip + 2 } :=
    step_alu_xor (by rw [ht1rip]; exact hc.p232) hv7 hv7
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 7 (some (aluF .xor v7 v7).1) with
        fl := (aluF .xor v7 v7).2, rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x234 := by
    rw [ht2]; show t1.rip + 2 = base + 0x234; rw [ht1rip]
  have ht2r7 : t2.reg 7 = some 0 := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg 7 = some 0
    rw [reg_set_self (by rw [ht1len]; decide), hxor]
  have ht2fr : ∀ i, i ≠ 0 → i ≠ 7 → t2.reg i = s.reg i := by
    intro i hi0 hi7
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg i = s.reg i
    rw [reg_set_other hi7]; exact ht1fr i hi0
  have ht2r0 : t2.reg 0 = some 9 := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg 0 = some 9
    rw [reg_set_other (by decide)]; exact ht1r0
  have ht2r13 : t2.reg 13 = some L := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).reg 13 = some L
    rw [reg_set_other (by decide)]; exact ht1r13
  have ht2r4 : t2.reg 4 = some sp1 := by
    rw [ht2fr 4 (by decide) (by decide)]; exact hr4
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2k : t2.k = s.k := by
    rw [ht2]
    show (t1.set 7 (some (aluF .xor v7 v7).1)).k = s.k
    exact ht1k
  -- 0x234: mov rsi, r13
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 6 (some L) with rip := t2.rip + 3 } :=
    step_mov (by rw [ht2rip]; exact hc.p234) ht2r13
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 6 (some L) with rip := t2.rip + 3 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x237 := by
    rw [ht3]; show t2.rip + 3 = base + 0x237; rw [ht2rip]
  have ht3r6 : t3.reg 6 = some L := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 6 = some L
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 6 → i ≠ 7 → t3.reg i = s.reg i := by
    intro i hi0 hi6 hi7
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi6]; exact ht2fr i hi0 hi7
  have ht3r0 : t3.reg 0 = some 9 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 0 = some 9
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r0
  have ht3r7 : t3.reg 7 = some 0 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 7 = some 0
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r7
  have ht3r4 : t3.reg 4 = some sp1 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).reg 4 = some sp1
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht2r4
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  have ht3k : t3.k = s.k := by
    rw [ht3]
    show ({ t2.set 6 (some L) with rip := t2.rip + 3 }).k = s.k
    exact ht2k
  -- 0x237: movI rdx, 3
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some (3 % W)) with rip := t3.rip + 5 } :=
    step_movI (by rw [ht3rip]; exact hc.p237)
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some (3 % W)) with rip := t3.rip + 5 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x23c := by
    rw [ht4]; show t3.rip + 5 = base + 0x23c; rw [ht3rip]
  have h3W : 3 % W = 3 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4r2 : t4.reg 2 = some 3 := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).reg 2 =
      some 3
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide), h3W]
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → i ≠ 7 → t4.reg i = s.reg i := by
    intro i hi0 hi2 hi6 hi7
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi2]; exact ht3fr i hi0 hi6 hi7
  have ht4r : ∀ j : Nat, j = 0 ∨ j = 4 ∨ j = 6 ∨ j = 7 →
      t4.reg j = t3.reg j := by
    intro j hj
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).reg j =
      t3.reg j
    rw [reg_with_rip, reg_set_other (by omega)]
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht3len
  have ht4k : t4.k = s.k := by
    rw [ht4]
    show ({ t3.set 2 (some (3 % W)) with rip := t3.rip + 5 }).k = s.k
    exact ht3k
  -- 0x23c: movI r10, 34
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 10 (some (34 % W)) with rip := t4.rip + 6 } :=
    step_movI (by rw [ht4rip]; exact hc.p23c)
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 10 (some (34 % W)) with rip := t4.rip + 6 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x242 := by
    rw [ht5]; show t4.rip + 6 = base + 0x242; rw [ht4rip]
  have h34W : 34 % W = 34 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht5r10 : t5.reg 10 = some 34 := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).reg 10 =
      some 34
    rw [reg_with_rip, reg_set_self (by rw [ht4len]; decide), h34W]
  have ht5fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t5.reg i = s.reg i := by
    intro i hi0 hi2 hi6 hi7 hi10
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi10]; exact ht4fr i hi0 hi2 hi6 hi7
  have ht5r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 →
      t5.reg j = t4.reg j := by
    intro j hj
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).reg j =
      t4.reg j
    rw [reg_with_rip, reg_set_other (by omega)]
  have ht5r4 : t5.reg 4 = some sp1 := by
    rw [ht5r 4 (by decide), ht4r 4 (by decide)]; exact ht3r4
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with
      rip := t4.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht4len
  have ht5k : t5.k = s.k := by
    rw [ht5]
    show ({ t4.set 10 (some (34 % W)) with rip := t4.rip + 6 }).k = s.k
    exact ht4k
  -- 0x242: pushI -1
  obtain ⟨m1, hm1raw⟩ : ∃ m, wr64 s.mem ((sp1 + W - 8 * (0 + 1)) % W)
      (18446744073709551615 % W) = some m :=
    Option.isSome_iff_exists.mp (hroom sp1 hr4 0 (by decide) _)
  have hm1 : wr64 s.mem ((sp1 + W - 8) % W) (18446744073709551615 % W) =
      some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp1 - 8) (18446744073709551615 % W) =
      some m1 := by
    rwa [hsp8mod] at hm1
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 4 (some ((sp1 + W - 8) % W)) with mem := m1, rip := t5.rip + 2 } := by
    have hm1' : wr64 t5.mem ((sp1 + W - 8) % W)
        (18446744073709551615 % W) = some m1 := by
      rw [ht5mem]; exact hm1
    exact step_pushI (by rw [ht5rip]; exact hc.p242) ht5r4 hm1'
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 4 (some ((sp1 + W - 8) % W)) with mem := m1, rip := t5.rip + 2 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x244 := by
    rw [ht6]; show t5.rip + 2 = base + 0x244; rw [ht5rip]
  have ht6r4 : t6.reg 4 = some (sp1 - 8) := by
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).reg 4 = some (sp1 - 8)
    rw [reg_set_self (by rw [ht5len]; decide), hsp8mod]
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
      t6.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi10
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi4]; exact ht5fr i hi0 hi2 hi6 hi7 hi10
  have ht6r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t6.reg j = t5.reg j := by
    intro j hj
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).reg j = t5.reg j
    rw [reg_set_other (by omega)]
  have ht6mem : t6.mem = m1 := by rw [ht6]
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).regs.length = 16
    rw [len_set]; exact ht5len
  have ht6k : t6.k = s.k := by
    rw [ht6]
    show (t5.set 4 (some ((sp1 + W - 8) % W))).k = s.k
    exact ht5k
  -- 0x244: pop r8
  have hImm : 18446744073709551615 % W = W - 1 := by
    have h : (18446744073709551615 : Nat) = W - 1 := by
      unfold W; decide
    rw [h]
    exact Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)
  have hrd64m1 : rd64 m1 (sp1 - 8) = some (W - 1) := by
    have h := rd64_wr64 hm1wr
    rwa [hImm, Nat.mod_eq_of_lt (by
      have hW : W = 2 ^ 64 := rfl; omega)] at h
  have e7 : X86Int.step stage1Im t6 =
      .ok { (t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8 (some (W - 1)) with
        rip := t6.rip + 2 } := by
    have hv : rd64 t6.mem (sp1 - 8) = some (W - 1) := by
      rw [ht6mem]; exact hrd64m1
    exact step_pop (by rw [ht6rip]; exact hc.p244) ht6r4 hv
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { (t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8 (some (W - 1)) with
        rip := t6.rip + 2 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x246 := by
    rw [ht7]; show t6.rip + 2 = base + 0x246; rw [ht6rip]
  have hspret : (sp1 - 8 + 8) % W = sp1 := by
    rw [show sp1 - 8 + 8 = sp1 by omega, Nat.mod_eq_of_lt hspW]
  have ht7r8 : t7.reg 8 = some (W - 1) := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg 8 = some (W - 1)
    rw [reg_set_self (by
      show 8 < (t6.set 4 (some ((sp1 - 8 + 8) % W))).regs.length
      rw [len_set, ht6len]; decide)]
  have ht7r4 : t7.reg 4 = some sp1 := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg 4 = some sp1
    rw [reg_set_other (by decide),
      reg_set_self (by rw [ht6len]; decide), hspret]
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 10 → t7.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi10
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg i = s.reg i
    rw [reg_set_other hi8, reg_set_other hi4]
    exact ht6fr i hi0 hi2 hi4 hi6 hi7 hi10
  have ht7r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 6 ∨ j = 7 ∨ j = 10 →
      t7.reg j = t6.reg j := by
    intro j hj
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).reg j = t6.reg j
    rw [reg_set_other (by omega), reg_set_other (by omega)]
  have ht7mem : t7.mem = m1 := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).mem = m1
    rw [mem_set, mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).regs.length = 16
    rw [len_set, len_set]; exact ht6len
  have ht7k : t7.k = s.k := by
    rw [ht7]
    show ((t6.set 4 (some ((sp1 - 8 + 8) % W))).set 8
      (some (W - 1))).k = s.k
    exact ht6k
  -- 0x246: xor r9, r9
  obtain ⟨v9, hv9⟩ : ∃ v, t7.reg 9 = some v := by
    obtain ⟨v, hv⟩ := hr9
    exact ⟨v, by
      rw [ht7fr 9 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide)]
      exact hv⟩
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t7.rip + 3 } :=
    step_alu_xor (by rw [ht7rip]; exact hc.p246) hv9 hv9
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 9 (some (aluF .xor v9 v9).1) with
        fl := (aluF .xor v9 v9).2, rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x249 := by
    rw [ht8]; show t7.rip + 3 = base + 0x249; rw [ht7rip]
  have ht8r9 : t8.reg 9 = some 0 := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).reg 9 = some 0
    rw [reg_set_self (by rw [ht7len]; decide), hxor]
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 8 →
      i ≠ 9 → i ≠ 10 → t8.reg i = s.reg i := by
    intro i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).reg i = s.reg i
    rw [reg_set_other hi9]
    exact ht7fr i hi0 hi2 hi4 hi6 hi7 hi8 hi10
  have ht8r : ∀ j : Nat, j = 0 ∨ j = 2 ∨ j = 4 ∨ j = 6 ∨ j = 7 ∨
      j = 8 ∨ j = 10 → t8.reg j = t7.reg j := by
    intro j hj
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).reg j = t7.reg j
    rw [reg_set_other (by omega)]
  have ht8mem : t8.mem = m1 := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).mem = m1
    rw [mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).regs.length = 16
    rw [len_set]; exact ht7len
  have ht8k : t8.k = s.k := by
    rw [ht8]
    show (t7.set 9 (some (aluF .xor v9 v9).1)).k = s.k
    exact ht7k
  have ht8r0 : t8.reg 0 = some 9 := by
    rw [ht8r 0 (Or.inl rfl), ht7r 0 (Or.inl rfl), ht6r 0 (Or.inl rfl),
      ht5r 0 (Or.inl rfl), ht4r 0 (Or.inl rfl)]
    exact ht3r0
  have ht8r7 : t8.reg 7 = some 0 := by
    rw [ht8r 7 (by decide), ht7r 7 (by decide), ht6r 7 (by decide),
      ht5r 7 (by decide), ht4r 7 (by decide)]
    exact ht3r7
  have ht8r6 : t8.reg 6 = some L := by
    rw [ht8r 6 (by decide), ht7r 6 (by decide), ht6r 6 (by decide),
      ht5r 6 (by decide), ht4r 6 (by decide)]
    exact ht3r6
  have ht8r2 : t8.reg 2 = some 3 := by
    rw [ht8r 2 (by decide), ht7r 2 (by decide), ht6r 2 (by decide),
      ht5r 2 (by decide)]
    exact ht4r2
  have ht8r10 : t8.reg 10 = some 34 := by
    rw [ht8r 10 (by decide), ht7r 10 (by decide), ht6r 10 (by decide)]
    exact ht5r10
  have ht8r8 : t8.reg 8 = some (W - 1) := by
    rw [ht8r 8 (by decide)]
    exact ht7r8
  have ht8r4 : t8.reg 4 = some sp1 := by
    rw [ht8r 4 (by decide)]
    exact ht7r4
  -- 0x249: syscall (mmap)
  have hw1 : wshape m1 = wshape s.mem := wrN_wshape _ _ _ hm1raw
  have hm1free : m1.free S (X86Env.pages L) = true := by
    rw [free_of_wshape hw1]; exact hfree
  have hplace : X86Env.place m1 (some (.at S)) (X86Env.pages L) =
      some (some S) :=
    place_at hSpage hSpos hSbound hm1free
  have hsc : X86Env.syscall t8.k t8.mem 9 0 L PROT_RW
      MAP_PRIVATE_ANON (W - 1) 0 =
      some (.ret S (Kern.next s.k).2
        (m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩])) := by
    rw [ht8k, ht8mem]
    exact syscall_mmap_exact hL hhead hplace
  have e9 : X86Int.step stage1Im t8 =
      .ok { ((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩], rip := t8.rip + 2 } :=
    step_syscall_ret (by rw [ht8rip]; exact hc.p249) ht8r0 ht8r7
      ht8r6 ht8r2 ht8r10 ht8r8 ht8r9 hsc
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { ((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11 none with k := (Kern.next s.k).2, mem := m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩], rip := t8.rip + 2 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x24b := by
    rw [ht9]; show t8.rip + 2 = base + 0x24b; rw [ht8rip]
  have ht9r0 : t9.reg 0 = some S := by
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).reg 0 = some S
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_self (by rw [ht8len]; decide)]
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t9.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).reg i = s.reg i
    rw [reg_set_other hi11, reg_set_other hi1, reg_set_other hi0]
    exact ht8fr i hi0 hi2 hi4 hi6 hi7 hi8 hi9 hi10
  have ht9r4 : t9.reg 4 = some sp1 := by
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).reg 4 = some sp1
    rw [reg_set_other (by decide), reg_set_other (by decide),
      reg_set_other (by decide)]
    exact ht8r4
  have ht9mem : t9.mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩] := by
    rw [ht9]
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (((t8.set 0 (some S)).set 1 (some (t8.rip + 2))).set 11
      none).regs.length = 16
    rw [len_set, len_set, len_set]; exact ht8len
  have ht9k : t9.k = (Kern.next s.k).2 := by
    rw [ht9]
  -- 0x24b: test rax, rax
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9 with fl := (aluF .and S S).2, rip := t9.rip + 3 } :=
    step_test (by rw [ht9rip]; exact hc.p24b) ht9r0 ht9r0
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9 with fl := (aluF .and S S).2, rip := t9.rip + 3 } := ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x24e := by
    rw [ht10]; show t9.rip + 3 = base + 0x24e; rw [ht9rip]
  have hmsb : msb S = false := by
    have hS63 : S < 2 ^ 63 := by
      have hpg : 0 ≤ X86Env.pages L := Nat.zero_le _
      have h47 : (2 : Nat) ^ 47 < 2 ^ 63 := by decide
      omega
    unfold msb
    rw [Nat.div_eq_of_lt hS63]
    rfl
  have ht10fl : X86Int.cond t10.fl 9 = some true := by
    have hsf : t10.fl.sf = some false := by
      rw [ht10]
      show (aluF .and S S).2.sf = some false
      have h1 : (aluF .and S S).2.sf = some (msb (aluF .and S S).1) := rfl
      rw [h1, aluF_and_fst_self, hmsb]
    simp [X86Int.cond, hsf]
  -- 0x24e: jns 0x255
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x255 } := by
    have h := step_jcc (im := stage1Im) (s := t10)
      (by rw [ht10rip]; exact hc.p24e) ht10fl
    simpa using h
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x255 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x255 := by rw [ht11]
  have ht11r0 : t11.reg 0 = some S := by
    have h10r0 : t10.reg 0 = some S := by
      rw [ht10]; show t9.reg 0 = some S; exact ht9r0
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).reg 0 = some S
    rw [reg_with_rip]; exact h10r0
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      i ≠ 8 → i ≠ 9 → i ≠ 10 → i ≠ 11 → t11.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    have h10i : t10.reg i = s.reg i := by
      rw [ht10]; show t9.reg i = s.reg i
      exact ht9fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).reg i = s.reg i
    rw [reg_with_rip]; exact h10i
  have ht11r4 : t11.reg 4 = some sp1 := by
    have h10r4 : t10.reg 4 = some sp1 := by
      rw [ht10]; show t9.reg 4 = some sp1; exact ht9r4
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).reg 4 = some sp1
    rw [reg_with_rip]; exact h10r4
  have ht11mem : t11.mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩] := by
    have h10m : t10.mem =
        m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩] := by
      rw [ht10]; show t9.mem = _; exact ht9mem
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩]
    rw [mem_with_rip]; exact h10m
  have ht11len : t11.regs.length = 16 := by
    have h10l : t10.regs.length = 16 := by
      rw [ht10]; show t9.regs.length = 16; exact ht9len
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).regs.length = 16
    rw [len_with_rip]; exact h10l
  have ht11k : t11.k = (Kern.next s.k).2 := by
    have h10k : t10.k = (Kern.next s.k).2 := by
      rw [ht10]; show t9.k = (Kern.next s.k).2; exact ht9k
    rw [ht11]
    show ({ t10 with rip := base + 0x255 }).k = (Kern.next s.k).2
    exact h10k
  -- 0x255: mov r11, rax
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 11 (some S) with rip := t11.rip + 3 } :=
    step_mov (by rw [ht11rip]; exact hc.p255) ht11r0
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 11 (some S) with rip := t11.rip + 3 } := ⟨_, rfl⟩
  -- compose the 12 steps
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  have hrun : run stage1Im 12 (Out.ok s) = Out.ok t12 := by
    rw [show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r1,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r2,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r3,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r4,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r5,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r6,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r7,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r8,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r9,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r10,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r11, r12]
  have hm1fr : ∀ x k, x + k ≤ sp1 - 8 ∨ sp1 ≤ x →
      m1.rdN x k = s.mem.rdN x k := by
    intro x k hdis
    apply rdN_wr64_other hm1wr
    rcases hdis with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)
  refine ⟨t12, m1, hrun, ?_, ?_, ?_, ?_, hm1fr, ?_, ?_, ?_, hw1⟩
  · rw [ht12]; show t11.rip + 3 = base + 0x258; rw [ht11rip]
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).reg 11 =
      some S
    rw [reg_with_rip, reg_set_self (by rw [ht11len]; decide)]
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).reg 4 =
      some sp1
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht11r4
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).mem =
      m1 ++ [⟨S, List.replicate (X86Env.pages L) 0, true⟩]
    rw [mem_with_rip, mem_set]; exact ht11mem
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
    rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi11]
    exact ht11fr i hi0 hi1 hi2 hi4 hi6 hi7 hi8 hi9 hi10 hi11
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht11len
  · rw [ht12]
    show ({ t11.set 11 (some S) with rip := t11.rip + 3 }).k =
      (Kern.next s.k).2
    exact ht11k

/-- One `pop rax` step with full frame. -/
theorem pop0_step {t : X86Int.St} {spx w : Nat}
    (hi : stage1Im t.rip = some (.pop 0, 1))
    (hr4 : t.reg 4 = some spx) (hspW : spx + 8 < W)
    (hv : rd64 t.mem spx = some w) (hlen : t.regs.length = 16) :
    ∃ t' : X86Int.St,
      run stage1Im 1 (Out.ok t) = Out.ok t' ∧
      t'.rip = t.rip + 1 ∧ t'.reg 0 = some w ∧
      t'.reg 4 = some (spx + 8) ∧
      (∀ i, i ≠ 0 → i ≠ 4 → t'.reg i = t.reg i) ∧
      t'.mem = t.mem ∧ t'.regs.length = 16 ∧ t'.k = t.k := by
  have e1 : X86Int.step stage1Im t =
      .ok { (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
        rip := t.rip + 1 } :=
    step_pop hi hr4 hv
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u =
      { (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
        rip := t.rip + 1 } := ⟨_, rfl⟩
  have hrun : run stage1Im 1 (Out.ok t) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 t, e1, ← ht1, run_zero]
  refine ⟨t1, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]
  · rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).reg 0 = some w
    rw [reg_with_rip, reg_set_self (by
      show 0 < (t.set 4 (some ((spx + 8) % W))).regs.length
      rw [len_set]; omega)]
  · rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).reg 4 = some (spx + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 0),
      reg_set_self (by omega), Nat.mod_eq_of_lt hspW]
  · intro i hi0 hi4
    rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).reg i = t.reg i
    rw [reg_with_rip, reg_set_other hi0, reg_set_other hi4]
  · rw [ht1]; rfl
  · rw [ht1]
    show ({ (t.set 4 (some ((spx + 8) % W))).set 0 (some w) with
      rip := t.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]; exact hlen
  · rw [ht1]; rfl

/-- Parser setup after the two mmaps: restore the cursor from the
saved slot, point `rdi` at the words table, reload `nwords` into
`r13` (masked to 32 bits), pop six initial-frame words to make
stack room for the checker's calls, and jump to the word loop. -/
theorem parser_setup_ok {sp1 c T B v w1 w2 w3 w4 w5 w6 : Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x258) (hr4 : s.reg 4 = some sp1)
    (hr14 : s.reg 14 = some T) (hr3 : s.reg 3 = some B)
    (hspW : sp1 + 56 < W) (hBW : B + 8 < W)
    (hcur : rd64 s.mem sp1 = some c)
    (hnw : rd64 s.mem (B + 8) = some v)
    (hp1 : rd64 s.mem (sp1 + 8) = some w1)
    (hp2 : rd64 s.mem (sp1 + 16) = some w2)
    (hp3 : rd64 s.mem (sp1 + 24) = some w3)
    (hp4 : rd64 s.mem (sp1 + 32) = some w4)
    (hp5 : rd64 s.mem (sp1 + 40) = some w5)
    (hp6 : rd64 s.mem (sp1 + 48) = some w6)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2c9 ∧ s'.reg 6 = some c ∧
      s'.reg 7 = some T ∧ s'.reg 13 = some (v % 2 ^ 32) ∧
      s'.reg 4 = some (sp1 + 56) ∧ s'.reg 0 = some w6 ∧
      s'.mem = s.mem ∧
      (∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x258: pop rsi (restore cursor)
  have e1 : X86Int.step stage1Im s =
      .ok { (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
        rip := s.rip + 1 } :=
    step_pop (by rw [hrip]; exact hc.p258) hr4 hcur
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
        rip := s.rip + 1 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x259 := by
    rw [ht1]; show s.rip + 1 = base + 0x259; rw [hrip]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).reg 6 = some c
    rw [reg_with_rip, reg_set_self (by
      show 6 < (s.set 4 (some ((sp1 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ht1r4 : t1.reg 4 = some (sp1 + 8) := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).reg 4 = some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 6),
      reg_set_self (hlen 4 (by decide)),
      Nat.mod_eq_of_lt (by omega)]
  have ht1fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t1.reg i = s.reg i := by
    intro i _ hi4 hi6 _ _
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi6, reg_set_other hi4]
  have ht1r14 : t1.reg 14 = some T := by
    rw [ht1fr 14 (by decide) (by decide) (by decide) (by decide)
      (by decide)]; exact hr14
  have ht1mem : t1.mem = s.mem := by rw [ht1]; rfl
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 6 (some c) with
      rip := s.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]; exact hlen16
  -- 0x259: mov rdi, r14
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 7 (some T) with rip := t1.rip + 3 } :=
    step_mov (by rw [ht1rip]; exact hc.p259) ht1r14
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 7 (some T) with rip := t1.rip + 3 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x25c := by
    rw [ht2]; show t1.rip + 3 = base + 0x25c; rw [ht1rip]
  have ht2r7 : t2.reg 7 = some T := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 7 = some T
    rw [reg_with_rip, reg_set_self (by rw [ht1len]; decide)]
  have ht2r3 : t2.reg 3 = some B := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 3 = some B
    rw [reg_with_rip, reg_set_other (by decide : (3 : Nat) ≠ 7),
      ht1fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide)]; exact hr3
  have ht2r6 : t2.reg 6 = some c := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 7)]
    exact ht1r6
  have ht2r4 : t2.reg 4 = some (sp1 + 8) := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 7)]
    exact ht1r4
  have ht2fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t2.reg i = s.reg i := by
    intro i hi0 hi4 hi6 hi7 hi13
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi7]
    exact ht1fr i hi0 hi4 hi6 hi7 hi13
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1.set 7 (some T) with rip := t1.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  -- 0x25c: ld r13, [rbx+8] (nwords field, 8-byte read inside header)
  have hld : rd64 t2.mem (ea B 8) = some v := by
    unfold ea
    rw [Nat.mod_eq_of_lt hBW, ht2mem]; exact hnw
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 13 (some v) with rip := t2.rip + 4 } :=
    step_ld (by rw [ht2rip]; exact hc.p25c) ht2r3 hld
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 13 (some v) with rip := t2.rip + 4 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x260 := by
    rw [ht3]; show t2.rip + 4 = base + 0x260; rw [ht2rip]
  have ht3r13 : t3.reg 13 = some v := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 13 = some v
    rw [reg_with_rip, reg_set_self (by rw [ht2len]; decide)]
  have ht3r6 : t3.reg 6 = some c := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 13)]
    exact ht2r6
  have ht3r7 : t3.reg 7 = some T := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 7 = some T
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 13)]
    exact ht2r7
  have ht3r4 : t3.reg 4 = some (sp1 + 8) := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 13)]
    exact ht2r4
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t3.reg i = s.reg i := by
    intro i hi0 hi4 hi6 hi7 hi13
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi13]
    exact ht2fr i hi0 hi4 hi6 hi7 hi13
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 13 (some v) with rip := t2.rip + 4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x260: mov32 r13, r13
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 } :=
    step_mov32 (by rw [ht3rip]; exact hc.p260) ht3r13
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x263 := by
    rw [ht4]; show t3.rip + 3 = base + 0x263; rw [ht3rip]
  have ht4r13 : t4.reg 13 = some (v % 2 ^ 32) := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 13 =
      some (v % 2 ^ 32)
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide)]
  have ht4r6 : t4.reg 6 = some c := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 6 =
      some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 13)]
    exact ht3r6
  have ht4r7 : t4.reg 7 = some T := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 7 =
      some T
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 13)]
    exact ht3r7
  have ht4r4 : t4.reg 4 = some (sp1 + 8) := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 13)]
    exact ht3r4
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 13 →
      t4.reg i = s.reg i := by
    intro i hi0 hi4 hi6 hi7 hi13
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi13]
    exact ht3fr i hi0 hi4 hi6 hi7 hi13
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with rip := t3.rip + 3 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 13 (some (v % 2 ^ 32)) with
      rip := t3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x263..0x268: six pop rax (initial-frame words, never read)
  obtain ⟨t5, r5, ht5rip, -, ht5r4, ht5fr, ht5mem, ht5len, ht5k⟩ :=
    pop0_step (t := t4) (by rw [ht4rip]; exact hc.p263) ht4r4
      (by omega) (by rw [ht4mem]; exact hp1) ht4len
  have ht5rip' : t5.rip = base + 0x264 := by rw [ht5rip, ht4rip]
  obtain ⟨t6, r6, ht6rip, -, ht6r4, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    pop0_step (t := t5) (by rw [ht5rip']; exact hc.p264) ht5r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 = sp1 + 16 := by omega
        rw [haddr, ht5mem, ht4mem]; exact hp2) ht5len
  have ht6rip' : t6.rip = base + 0x265 := by rw [ht6rip, ht5rip']
  obtain ⟨t7, r7, ht7rip, -, ht7r4, ht7fr, ht7mem, ht7len, ht7k⟩ :=
    pop0_step (t := t6) (by rw [ht6rip']; exact hc.p265) ht6r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 = sp1 + 24 := by omega
        rw [haddr, ht6mem, ht5mem, ht4mem]; exact hp3) ht6len
  have ht7rip' : t7.rip = base + 0x266 := by rw [ht7rip, ht6rip']
  obtain ⟨t8, r8, ht8rip, -, ht8r4, ht8fr, ht8mem, ht8len, ht8k⟩ :=
    pop0_step (t := t7) (by rw [ht7rip']; exact hc.p266) ht7r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 + 8 = sp1 + 32 := by omega
        rw [haddr, ht7mem, ht6mem, ht5mem, ht4mem]; exact hp4) ht7len
  have ht8rip' : t8.rip = base + 0x267 := by rw [ht8rip, ht7rip']
  obtain ⟨t9, r9, ht9rip, -, ht9r4, ht9fr, ht9mem, ht9len, ht9k⟩ :=
    pop0_step (t := t8) (by rw [ht8rip']; exact hc.p267) ht8r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 + 8 + 8 = sp1 + 40 := by omega
        rw [haddr, ht8mem, ht7mem, ht6mem, ht5mem, ht4mem]
        exact hp5) ht8len
  have ht9rip' : t9.rip = base + 0x268 := by rw [ht9rip, ht8rip']
  obtain ⟨t10, r10, ht10rip, ht10r0, ht10r4, ht10fr, ht10mem, ht10len,
      ht10k⟩ :=
    pop0_step (t := t9) (by rw [ht9rip']; exact hc.p268) ht9r4
      (by omega) (by
        have haddr : sp1 + 8 + 8 + 8 + 8 + 8 + 8 = sp1 + 48 := by omega
        rw [haddr, ht9mem, ht8mem, ht7mem, ht6mem, ht5mem, ht4mem]
        exact hp6) ht9len
  have ht10rip' : t10.rip = base + 0x269 := by rw [ht10rip, ht9rip']
  -- registers 6, 7, 13 survive the six pops; frame composes
  have fr10 : ∀ i, i ≠ 0 → i ≠ 4 → t10.reg i = t4.reg i := by
    intro i hi0 hi4
    rw [ht10fr i hi0 hi4, ht9fr i hi0 hi4, ht8fr i hi0 hi4,
      ht7fr i hi0 hi4, ht6fr i hi0 hi4, ht5fr i hi0 hi4]
  have ht10r6 : t10.reg 6 = some c := by
    rw [fr10 6 (by decide) (by decide)]; exact ht4r6
  have ht10r7 : t10.reg 7 = some T := by
    rw [fr10 7 (by decide) (by decide)]; exact ht4r7
  have ht10r13 : t10.reg 13 = some (v % 2 ^ 32) := by
    rw [fr10 13 (by decide) (by decide)]; exact ht4r13
  have ht10r4' : t10.reg 4 = some (sp1 + 56) := by
    have haddr : sp1 + 8 + 8 + 8 + 8 + 8 + 8 + 8 = sp1 + 56 := by omega
    rw [ht10r4, haddr]
  have ht10mem' : t10.mem = s.mem := by
    rw [ht10mem, ht9mem, ht8mem, ht7mem, ht6mem, ht5mem]; exact ht4mem
  have ht10k' : t10.k = s.k := by
    rw [ht10k, ht9k, ht8k, ht7k, ht6k, ht5k]; exact ht4k
  -- 0x269: jmp word loop
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10 with rip := base + 0x2c9 } :=
    step_jmp (by rw [ht10rip']; exact hc.p269)
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10 with rip := base + 0x2c9 } := ⟨_, rfl⟩
  -- compose the 11 steps
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  have hrun : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
    rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10, r11]
  refine ⟨t11, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht11]
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 6 = some c
    rw [reg_with_rip]; exact ht10r6
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 7 = some T
    rw [reg_with_rip]; exact ht10r7
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 13 = some (v % 2 ^ 32)
    rw [reg_with_rip]; exact ht10r13
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 4 = some (sp1 + 56)
    rw [reg_with_rip]; exact ht10r4'
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg 0 = some w6
    rw [reg_with_rip]; exact ht10r0
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).mem = s.mem
    exact ht10mem'
  · intro i hi0 hi4 hi6 hi7 hi13
    rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).reg i = s.reg i
    rw [reg_with_rip, fr10 i hi0 hi4]
    exact ht4fr i hi0 hi4 hi6 hi7 hi13
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).regs.length = 16
    rw [len_with_rip]; exact ht10len
  · rw [ht11]
    show ({ t10 with rip := base + 0x2c9 }).k = s.k
    exact ht10k'

/-- Word-loop entry: copy `nwords` from `r13` into the loop
counter `rcx`. -/
theorem parser_words_init {nw : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x2c9) (hr13 : s.reg 13 = some nw)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some nw ∧
      (∀ i, i ≠ 1 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 1 (some nw) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p2c9) hr13
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 1 (some nw) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  refine ⟨t1, r1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]; show s.rip + 3 = base + 0x2cc; rw [hrip]
  · rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).reg 1 = some nw
    rw [reg_with_rip, reg_set_self (by omega)]
  · intro i hi
    rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  · rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  · rw [ht1]
    show ({ s.set 1 (some nw) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  · rw [ht1]; exact rfl

/-- Word-loop head with words remaining: enter the body. -/
theorem parser_words_head_pos {n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x2cc) (hr1 : s.reg 1 = some n)
    (hn : n ≠ 0) (hnW : n < W) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2d6 ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and n n).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p2cc) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and n n).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x2cf := by
    rw [ht1]; show s.rip + 3 = base + 0x2cf; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some true := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .and n n).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [Nat.mod_eq_of_lt hnW, hn]
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x2d6 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p2cf) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x2d6 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]
  · intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).reg i = s.reg i
    rw [reg_with_rip]
    rw [ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).mem = s.mem
    rw [ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x2d6 }).k = s.k
    rw [ht1]

/-- Word-loop head with no words remaining: jump to the trailing
check. -/
theorem parser_words_head_zero {s : X86Int.St}
    (hrip : s.rip = base + 0x2cc) (hr1 : s.reg 1 = some 0) :
    ∃ s' : X86Int.St,
      run stage1Im 3 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x419 ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p2cc) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x2cf := by
    rw [ht1]; show s.rip + 3 = base + 0x2cf; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some false := by
    have hzf : t1.fl.zf = some true := by
      rw [ht1]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p2cf) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x2d1 := by
    rw [ht2]; show t1.rip + 2 = base + 0x2d1; rw [ht1rip]
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x419 } :=
    step_jmp (by rw [ht2rip]; exact hc.p2d1)
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x419 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have hrun : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
    rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, r3]
  refine ⟨t3, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht3]
  · intro i
    rw [ht3]
    show ({ t2 with rip := base + 0x419 }).reg i = s.reg i
    rw [reg_with_rip, ht2]
    show ({ t1 with rip := t1.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht3]
    show ({ t2 with rip := base + 0x419 }).mem = s.mem
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).mem = s.mem
    rw [ht1]
  · rw [ht3]
    show ({ t2 with rip := base + 0x419 }).regs.length = s.regs.length
    rw [len_with_rip, ht2]
    show ({ t1 with rip := t1.rip + 2 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht3]
    show ({ t2 with rip := base + 0x419 }).k = s.k
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).k = s.k
    rw [ht1]

/-- One table-slot store `st [rdi+off], r` with full frame. -/
theorem store_slot {site off slen p v r : Nat} {m : Mem} {s : X86Int.St}
    (hst : stage1Im site = some (.st 7 off r, slen))
    (hrip : s.rip = site) (hr7 : s.reg 7 = some p)
    (hrv : s.reg r = some v) (hpW : p + off < W)
    (hm : wr64 s.mem (p + off) v = some m) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧
      s'.rip = site + slen ∧ s'.mem = m ∧
      rd64 s'.mem (p + off) = some (v % W) ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, p + off + 8 ≤ x ∨ x + k ≤ p + off →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hea : ea p off = p + off := by
    unfold ea
    rw [Nat.mod_eq_of_lt hpW]
  have hm' : wr64 s.mem (ea p off) v = some m := by rwa [hea]
  have e1 : X86Int.step stage1Im s =
      .ok { s with mem := m, rip := s.rip + slen } :=
    step_st (by rw [hrip]; exact hst) hr7 hrv hm'
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with mem := m, rip := s.rip + slen } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have ht1mem : t1.mem = m := by rw [ht1]
  refine ⟨t1, r1, ?_, ht1mem, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]; show s.rip + slen = site + slen; rw [hrip]
  · rw [ht1mem]; exact rd64_wr64 hm
  · intro i
    rw [ht1]
    show ({ s with mem := m, rip := s.rip + slen }).reg i = s.reg i
    exact rfl
  · rw [ht1]
  · rw [ht1]
  · rw [ht1mem]
    exact wrN_wshape _ _ _ hm
  · intro x k hd
    rw [ht1mem]
    exact rdN_wr64_other hm hd

/-- One word field: `call RDC` then store `rax` at `[rdi+off]`. -/
theorem field_store {site off slen c e sp p : Nat} {bs4 : List Nat}
    {s : X86Int.St}
    (hcall : stage1Im site = some (.call aRDC, 5))
    (hst : stage1Im (site + 5) = some (.st 7 off 0, slen))
    (hsiteW : site + 5 < W)
    (hrip : s.rip = site) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 4 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : s.mem.rdN c 4 = some bs4) (hroom : StackRoom s 2)
    (hr4 : s.reg 4 = some sp) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hdisj : c + 4 ≤ sp - 16 ∨ sp ≤ c)
    (hr7 : s.reg 7 = some p) (hpW : p + off < W)
    (hdisjp : p + off + 8 ≤ c ∨ c + 4 ≤ p + off)
    (hwr : (wr64 s.mem (p + off) (leVal bs4)).isSome) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = site + 5 + slen ∧
      s'.reg 6 = some (c + 4) ∧ s'.reg 4 = some sp ∧
      rd64 s'.mem (p + off) = some (leVal bs4 % W) ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      s'.mem.rdN c 4 = some bs4 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + off + 8 ≤ x ∨ x + k ≤ p + off) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  obtain ⟨n1, s1, hrun1, hs1rip, hs1r0, hs1r6, hs1r4, hs1fr, hs1len,
      hs1memfr, hs1rd, hs1k, hs1memwr⟩ :=
    call_rdc_ok hcall hsiteW hrip hr6 hr15 hce hcW heW
      (by rw [hrd]; exact Option.isSome_some) hroom hr4 hsp16 hspW
      hlen16 hdisj
  obtain ⟨m1, m2, hm1, hm2, hs1mem⟩ := hs1memwr
  have hs1r0v : s1.reg 0 = some (leVal bs4) := by
    rw [hs1r0, hrd]; rfl
  have hs1r7 : s1.reg 7 = some p := by
    rw [hs1fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hr7
  have hwr1 : (wr64 m1 (p + off) (leVal bs4)).isSome :=
    wr64_some_of_wr64 hm1 hwr
  have hwr2 : (wr64 s1.mem (p + off) (leVal bs4)).isSome := by
    rw [hs1mem]
    exact wr64_some_of_wr64 hm2 hwr1
  obtain ⟨m3, hm3⟩ : ∃ m, wr64 s1.mem (p + off) (leVal bs4) = some m :=
    Option.isSome_iff_exists.mp hwr2
  obtain ⟨s2, hrun2, hs2rip, hs2mem, hs2slot, hs2fr, hs2len, hs2k,
      hs2sh, hs2memfr⟩ :=
    store_slot hst hs1rip hs1r7 hs1r0v hpW hm3
  have hsh : wshape s2.mem = wshape s.mem := by
    rw [hs2sh, hs1mem]
    have h2 : wshape m2 = wshape m1 := wrN_wshape _ _ _ hm2
    have h1 : wshape m1 = wshape s.mem := wrN_wshape _ _ _ hm1
    rw [h2, h1]
  have hroom2 : StackRoom s2 2 := by
    apply StackRoom_of_wshape hsh _ hroom
    rw [hs2fr 4]
    rw [hs1r4, hr4]
  have hrun : run stage1Im (n1 + 1) (Out.ok s) = Out.ok s2 := by
    rw [run_add stage1Im n1 1, hrun1, hrun2]
  refine ⟨n1 + 1, s2, hrun, hs2rip, ?_, ?_, ?_, ?_, ?_, ?_, hsh, hroom2,
    ?_, ?_⟩
  · rw [hs2fr 6]; exact hs1r6
  · rw [hs2fr 4]; exact hs1r4
  · exact hs2slot
  · intro i hi0 hi2 hi4 hi6
    rw [hs2fr i]
    exact hs1fr i hi0 hi2 hi4 hi6
  · rw [hs2len]; exact hs1len
  · rw [hs2k]; exact hs1k
  · rw [hs2memfr c 4 hdisjp, hs1rd, hrd]
  · intro x k hd1 hd2
    rw [hs2memfr x k hd2]
    exact hs1memfr x k hd1

/-- The five header fields of one word: five `RDC` reads stored
at entry slots 0, 8, 16, 24, 32, then the cursor (code pointer)
stored at slot 40.  Ends at the code-loop setup. -/
theorem parser_word_fields {c e sp p : Nat} {bs : List Nat}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr7 : s.reg 7 = some p)
    (hr4 : s.reg 4 = some sp)
    (h20 : s.mem.rdN c 20 = some bs) (hce : c + 20 ≤ e)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hEB : p + 64 ≤ c ∨ e ≤ p)
    (hSB : e ≤ sp - 16 ∨ sp ≤ c)
    (hES : p + 64 ≤ sp - 16 ∨ sp ≤ p)
    (hwr : ∀ (off v : Nat), off < 64 →
      (wr64 s.mem (p + off) v).isSome) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x306 ∧ s'.reg 6 = some (c + 20) ∧
      s'.reg 4 = some sp ∧ s'.reg 7 = some p ∧
      rd64 s'.mem p = some (leVal (bs.take 4) % W) ∧
      rd64 s'.mem (p + 8) = some (leVal ((bs.drop 4).take 4) % W) ∧
      rd64 s'.mem (p + 16) = some (leVal ((bs.drop 8).take 4) % W) ∧
      rd64 s'.mem (p + 24) = some (leVal ((bs.drop 12).take 4) % W) ∧
      rd64 s'.mem (p + 32) = some (leVal ((bs.drop 16).take 4) % W) ∧
      rd64 s'.mem (p + 40) = some ((c + 20) % W) ∧
      (∀ i, i ≠ 0 → i ≠ 2 → i ≠ 4 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      s'.mem.rdN c 20 = some bs ∧
      (∀ (off v : Nat), off < 64 →
        (wr64 s'.mem (p + off) v).isSome) ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 ≤ x ∨ x + k ≤ p) → s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hbW : base + 0x400 < W := by decide
  -- field slices of the 20 header bytes
  have hf0 : s.mem.rdN c 4 = some (bs.take 4) := by
    have h := rdN_slice h20 (show 0 + 4 ≤ 20 by decide)
    simpa using h
  -- slot 0
  obtain ⟨n1, s1, hrun1, hs1rip, hs1r6, hs1r4, hs1slot, hs1fr, hs1len,
      hs1k, hs1sh, hs1room, hs1rd, hs1memfr⟩ :=
    field_store (site := base + 0x2d6) (off := 0) (slen := 3)
      hc.p2d6 hc.p2db (Nat.lt_of_le_of_lt (by omega) hbW) hrip hr6 hr15 (by omega) hcW heW
      hf0 hroom hr4 hsp16 hspW hlen16 (by omega) hr7 (by omega)
      (by omega) (by
        have h := hwr 0 (leVal (bs.take 4)) (by decide)
        simpa using h)
  have hs1rip' : s1.rip = base + 0x2de := hs1rip
  have hs1r15 : s1.reg 15 = some e := by
    rw [hs1fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have hs1r7 : s1.reg 7 = some p := by
    rw [hs1fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hr7
  have hs1h20 : s1.mem.rdN c 20 = some bs := by
    rw [hs1memfr c 20 (by omega) (by omega)]; exact h20
  have hs1wr : ∀ (off v : Nat), off < 64 →
      (wr64 s1.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s.mem s1.mem hs1sh _
      (hwr off v hoff)
  have hf1 : s1.mem.rdN (c + 4) 4 = some ((bs.drop 4).take 4) :=
    rdN_slice hs1h20 (show 4 + 4 ≤ 20 by decide)
  -- slot 8
  obtain ⟨n2, s2, hrun2, hs2rip, hs2r6, hs2r4, hs2slot, hs2fr, hs2len,
      hs2k, hs2sh, hs2room, hs2rd, hs2memfr⟩ :=
    field_store (site := base + 0x2de) (off := 8) (slen := 4)
      hc.p2de hc.p2e3 (Nat.lt_of_le_of_lt (by omega) hbW) hs1rip' hs1r6 hs1r15 (by omega)
      (by omega) heW hf1 hs1room hs1r4 hsp16 hspW hs1len (by omega)
      hs1r7 (by omega) (by omega) (hs1wr 8 _ (by decide))
  have hs2rip' : s2.rip = base + 0x2e7 := hs2rip
  have hs2r15 : s2.reg 15 = some e := by
    rw [hs2fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs1r15
  have hs2r7 : s2.reg 7 = some p := by
    rw [hs2fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs1r7
  have hs2h20 : s2.mem.rdN c 20 = some bs := by
    rw [hs2memfr c 20 (by omega) (by omega)]; exact hs1h20
  have hs2wr : ∀ (off v : Nat), off < 64 →
      (wr64 s2.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s1.mem s2.mem hs2sh _
      (hs1wr off v hoff)
  have hs2slot0 : rd64 s2.mem p = some (leVal (bs.take 4) % W) := by
    have h : s2.mem.rdN p 8 = s1.mem.rdN p 8 :=
      hs2memfr p 8 (by omega) (by omega)
    unfold rd64
    rw [h]
    have h1 : rd64 s1.mem (p + 0) = some (leVal (bs.take 4) % W) :=
      hs1slot
    unfold rd64 at h1
    simpa using h1
  have hf2 : s2.mem.rdN (c + 8) 4 = some ((bs.drop 8).take 4) :=
    rdN_slice hs2h20 (show 8 + 4 ≤ 20 by decide)
  have hs2r6' : s2.reg 6 = some (c + 8) := by
    have e1 : c + 4 + 4 = c + 8 := by omega
    rwa [e1] at hs2r6
  -- slot 16
  obtain ⟨n3, s3, hrun3, hs3rip, hs3r6, hs3r4, hs3slot, hs3fr, hs3len,
      hs3k, hs3sh, hs3room, hs3rd, hs3memfr⟩ :=
    field_store (site := base + 0x2e7) (off := 16) (slen := 4)
      hc.p2e7 hc.p2ec (Nat.lt_of_le_of_lt (by omega) hbW) hs2rip' hs2r6' hs2r15 (by omega)
      (by omega) heW hf2 hs2room hs2r4 hsp16 hspW hs2len (by omega)
      hs2r7 (by omega) (by omega) (hs2wr 16 _ (by decide))
  have hs3rip' : s3.rip = base + 0x2f0 := hs3rip
  have hs3r15 : s3.reg 15 = some e := by
    rw [hs3fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r15
  have hs3r7 : s3.reg 7 = some p := by
    rw [hs3fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs2r7
  have hs3h20 : s3.mem.rdN c 20 = some bs := by
    rw [hs3memfr c 20 (by omega) (by omega)]; exact hs2h20
  have hs3wr : ∀ (off v : Nat), off < 64 →
      (wr64 s3.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s2.mem s3.mem hs3sh _
      (hs2wr off v hoff)
  have hs3slot0 : rd64 s3.mem p = some (leVal (bs.take 4) % W) := by
    unfold rd64
    rw [hs3memfr p 8 (by omega) (by omega)]
    have h := hs2slot0
    unfold rd64 at h
    exact h
  have hs3slot8 : rd64 s3.mem (p + 8) =
      some (leVal ((bs.drop 4).take 4) % W) := by
    unfold rd64
    rw [hs3memfr (p + 8) 8 (by omega) (by omega)]
    have h := hs2slot
    unfold rd64 at h
    exact h
  have hf3 : s3.mem.rdN (c + 12) 4 = some ((bs.drop 12).take 4) :=
    rdN_slice hs3h20 (show 12 + 4 ≤ 20 by decide)
  have hs3r6' : s3.reg 6 = some (c + 12) := by
    have e1 : c + 8 + 4 = c + 12 := by omega
    rwa [e1] at hs3r6
  -- slot 24
  obtain ⟨n4, s4, hrun4, hs4rip, hs4r6, hs4r4, hs4slot, hs4fr, hs4len,
      hs4k, hs4sh, hs4room, hs4rd, hs4memfr⟩ :=
    field_store (site := base + 0x2f0) (off := 24) (slen := 4)
      hc.p2f0 hc.p2f5 (Nat.lt_of_le_of_lt (by omega) hbW) hs3rip' hs3r6' hs3r15 (by omega)
      (by omega) heW hf3 hs3room hs3r4 hsp16 hspW hs3len (by omega)
      hs3r7 (by omega) (by omega) (hs3wr 24 _ (by decide))
  have hs4rip' : s4.rip = base + 0x2f9 := hs4rip
  have hs4r15 : s4.reg 15 = some e := by
    rw [hs4fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hs3r15
  have hs4r7 : s4.reg 7 = some p := by
    rw [hs4fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs3r7
  have hs4h20 : s4.mem.rdN c 20 = some bs := by
    rw [hs4memfr c 20 (by omega) (by omega)]; exact hs3h20
  have hs4wr : ∀ (off v : Nat), off < 64 →
      (wr64 s4.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s3.mem s4.mem hs4sh _
      (hs3wr off v hoff)
  have hs4slot0 : rd64 s4.mem p = some (leVal (bs.take 4) % W) := by
    unfold rd64
    rw [hs4memfr p 8 (by omega) (by omega)]
    have h := hs3slot0
    unfold rd64 at h
    exact h
  have hs4slot8 : rd64 s4.mem (p + 8) =
      some (leVal ((bs.drop 4).take 4) % W) := by
    unfold rd64
    rw [hs4memfr (p + 8) 8 (by omega) (by omega)]
    have h := hs3slot8
    unfold rd64 at h
    exact h
  have hs4slot16 : rd64 s4.mem (p + 16) =
      some (leVal ((bs.drop 8).take 4) % W) := by
    unfold rd64
    rw [hs4memfr (p + 16) 8 (by omega) (by omega)]
    have h := hs3slot
    unfold rd64 at h
    exact h
  have hf4 : s4.mem.rdN (c + 16) 4 = some ((bs.drop 16).take 4) :=
    rdN_slice hs4h20 (show 16 + 4 ≤ 20 by decide)
  have hs4r6' : s4.reg 6 = some (c + 16) := by
    have e1 : c + 12 + 4 = c + 16 := by omega
    rwa [e1] at hs4r6
  -- slot 32
  obtain ⟨n5, s5, hrun5, hs5rip, hs5r6, hs5r4, hs5slot, hs5fr, hs5len,
      hs5k, hs5sh, hs5room, hs5rd, hs5memfr⟩ :=
    field_store (site := base + 0x2f9) (off := 32) (slen := 4)
      hc.p2f9 hc.p2fe (Nat.lt_of_le_of_lt (by omega) hbW) hs4rip' hs4r6' hs4r15 (by omega)
      (by omega) heW hf4 hs4room hs4r4 hsp16 hspW hs4len (by omega)
      hs4r7 (by omega) (by omega) (hs4wr 32 _ (by decide))
  have hs5rip' : s5.rip = base + 0x302 := hs5rip
  have hs5r7 : s5.reg 7 = some p := by
    rw [hs5fr 7 (by decide) (by decide) (by decide) (by decide)]
    exact hs4r7
  have hs5h20 : s5.mem.rdN c 20 = some bs := by
    rw [hs5memfr c 20 (by omega) (by omega)]; exact hs4h20
  have hs5wr : ∀ (off v : Nat), off < 64 →
      (wr64 s5.mem (p + off) v).isSome := by
    intro off v hoff
    exact wrN_some_of_wshape _ s4.mem s5.mem hs5sh _
      (hs4wr off v hoff)
  have hs5r6' : s5.reg 6 = some (c + 20) := by
    have e1 : c + 16 + 4 = c + 20 := by omega
    rwa [e1] at hs5r6
  have slot_at : ∀ {off : Nat} {w : Nat},
      rd64 s4.mem (p + off) = some w → off + 8 ≤ 32 →
      rd64 s5.mem (p + off) = some w := by
    intro off w h hoff
    unfold rd64
    rw [hs5memfr (p + off) 8 (by omega) (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  -- slot 40: store the cursor (code pointer)
  obtain ⟨m6, hm6⟩ : ∃ m, wr64 s5.mem (p + 40) (c + 20) = some m :=
    Option.isSome_iff_exists.mp (hs5wr 40 _ (by decide))
  obtain ⟨s6, hrun6, hs6rip, hs6mem, hs6slot, hs6fr, hs6len, hs6k,
      hs6sh, hs6memfr⟩ :=
    store_slot (site := base + 0x302) (off := 40) (slen := 4)
      (r := 6) hc.p302 hs5rip' hs5r7 hs5r6' (by omega) hm6
  have hs6sh' : wshape s6.mem = wshape s.mem := by
    rw [hs6sh, hs5sh, hs4sh, hs3sh, hs2sh, hs1sh]
  have hrun : run stage1Im (n1 + n2 + n3 + n4 + n5 + 1) (Out.ok s) =
      Out.ok s6 := by
    rw [show n1 + n2 + n3 + n4 + n5 + 1 =
        n1 + (n2 + (n3 + (n4 + (n5 + 1)))) by omega,
      run_add stage1Im n1 _, hrun1, run_add stage1Im n2 _, hrun2,
      run_add stage1Im n3 _, hrun3, run_add stage1Im n4 _, hrun4,
      run_add stage1Im n5 1, hrun5, hrun6]
  have slot6 : ∀ {off : Nat} {w : Nat},
      rd64 s5.mem (p + off) = some w → off + 8 ≤ 40 →
      rd64 s6.mem (p + off) = some w := by
    intro off w h hoff
    unfold rd64
    rw [hs6memfr (p + off) 8 (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  have hslot0 : rd64 s6.mem p = some (leVal (bs.take 4) % W) := by
    have h5 : rd64 s5.mem (p + 0) = some (leVal (bs.take 4) % W) :=
      slot_at (off := 0) (by simpa using hs4slot0) (by decide)
    have h6 := slot6 (off := 0) h5 (by decide)
    simpa using h6
  have hroom6 : StackRoom s6 2 := by
    apply StackRoom_of_wshape hs6sh' _ hroom
    rw [hs6fr 4, hs5r4, hr4]
  refine ⟨n1 + n2 + n3 + n4 + n5 + 1, s6, hrun, hs6rip, ?_, ?_, ?_,
    hslot0, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hs6sh', hroom6, ?_, ?_, ?_⟩
  · rw [hs6fr 6]; exact hs5r6'
  · rw [hs6fr 4]; exact hs5r4
  · rw [hs6fr 7]; exact hs5r7
  · exact slot6 (slot_at hs4slot8 (by decide)) (by decide)
  · exact slot6 (slot_at hs4slot16 (by decide)) (by decide)
  · have h5 : rd64 s5.mem (p + 24) =
        some (leVal ((bs.drop 12).take 4) % W) :=
      slot_at hs4slot (by decide)
    exact slot6 h5 (by decide)
  · exact slot6 hs5slot (by decide)
  · exact hs6slot
  · intro i hi0 hi2 hi4 hi6
    rw [hs6fr i, hs5fr i hi0 hi2 hi4 hi6, hs4fr i hi0 hi2 hi4 hi6,
      hs3fr i hi0 hi2 hi4 hi6, hs2fr i hi0 hi2 hi4 hi6,
      hs1fr i hi0 hi2 hi4 hi6]
  · rw [hs6len]; exact hs5len
  · rw [hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]
  · rw [hs6memfr c 20 (by omega)]; exact hs5h20
  · intro off v hoff
    exact wrN_some_of_wshape _ s.mem s6.mem hs6sh' _
      (hwr off v hoff)
  · intro x k hd1 hd2
    rw [hs6memfr x k (by omega), hs5memfr x k hd1 (by omega),
      hs4memfr x k hd1 (by omega), hs3memfr x k hd1 (by omega),
      hs2memfr x k hd1 (by omega), hs1memfr x k hd1 (by omega)]

/-- Code-loop bounds check and opcode load: from the body entry
to the length dispatch, with the opcode byte in `rax`. -/
theorem parser_code_bounds {c e op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x315) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hce : c + 1 ≤ e)
    (hcW : c < W) (heW : e < W)
    (hrd : s.mem.rd c = some op) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 6 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x330 ∧ s'.reg 0 = some op ∧
      s'.reg 6 = some c ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  -- 0x315: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p315) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x318 := by
    rw [ht1]; show s.rip + 3 = base + 0x318; rw [hrip]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1fr : ∀ i, i ≠ 2 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- 0x318: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p318) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht2rip : t2.rip = base + 0x31b := by
    rw [ht2]; show t1.rip + 3 = base + 0x31b; rw [ht1rip]
  have ht2r2 : t2.reg 2 = some (e - c) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht1len]; decide), hsub]
  have ht2fr : ∀ i, i ≠ 2 → t2.reg i = s.reg i := by
    intro i hi
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht1fr i hi
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have ht2fl : X86Int.cond t2.fl 3 = some true := by
    have hcf : t2.fl.cf = some false := by
      rw [ht2]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  -- 0x31b: jae 0x322
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x322 } := by
    have h := step_jcc (im := stage1Im) (s := t2)
      (by rw [ht2rip]; exact hc.p31b) ht2fl
    simpa using h
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x322 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x322 := by rw [ht3]
  have ht3r2 : t3.reg 2 = some (e - c) := by
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).reg 2 = some (e - c)
    rw [reg_with_rip]; exact ht2r2
  have ht3fr : ∀ i, i ≠ 2 → t3.reg i = s.reg i := by
    intro i hi
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht2fr i hi
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).mem = s.mem
    rw [mem_with_rip]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2 with rip := base + 0x322 }).regs.length = 16
    rw [len_with_rip]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x322: cmp rdx, 1
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with fl := (aluF .sub (e - c) (1 % W)).2, rip := t3.rip + 4 } :=
    step_cmpI (by rw [ht3rip]; exact hc.p322) ht3r2
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with fl := (aluF .sub (e - c) (1 % W)).2, rip := t3.rip + 4 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x326 := by
    rw [ht4]; show t3.rip + 4 = base + 0x326; rw [ht3rip]
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  have ht4fl : X86Int.cond t4.fl 3 = some true := by
    have hcf : t4.fl.cf = some false := by
      rw [ht4]
      show (aluF .sub (e - c) (1 % W)).2.cf = some false
      rw [h1W]; exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht4fr : ∀ i, i ≠ 2 → t4.reg i = s.reg i := by
    intro i hi
    rw [ht4]
    show t3.reg i = s.reg i
    exact ht3fr i hi
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show t3.mem = s.mem
    exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]; show t3.regs.length = 16; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x326: jae 0x32d
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4 with rip := base + 0x32d } := by
    have h := step_jcc (im := stage1Im) (s := t4)
      (by rw [ht4rip]; exact hc.p326) ht4fl
    simpa using h
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4 with rip := base + 0x32d } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x32d := by rw [ht5]
  have ht5r6 : t5.reg 6 = some c := by
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).reg 6 = some c
    rw [reg_with_rip, ht4fr 6 (by decide)]; exact hr6
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).reg i = s.reg i
    rw [reg_with_rip]; exact ht4fr i hi
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).mem = s.mem
    rw [mem_with_rip]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4 with rip := base + 0x32d }).regs.length = 16
    rw [len_with_rip]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  -- 0x32d: ld8 rax, [rsi]
  have hld : t5.mem.rd (ea c 0) = some op := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hcW, ht5mem]; exact hrd
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some op) with rip := t5.rip + 3 } :=
    step_ld8 (by rw [ht5rip]; exact hc.p32d) ht5r6 hld
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some op) with rip := t5.rip + 3 } := ⟨_, rfl⟩
  -- compose
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have hrun : run stage1Im 6 (Out.ok s) = Out.ok t6 := by
    rw [show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r1,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r2,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r3,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r4,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r5, r6]
  refine ⟨t6, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht6]; show t5.rip + 3 = base + 0x330; rw [ht5rip]
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).reg 0 = some op
    rw [reg_with_rip, reg_set_self (by rw [ht5len]; decide)]
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht5r6
  · intro i hi0 hi2
    rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht5fr i hi2
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht5mem
  · rw [ht6]
    show ({ t5.set 0 (some op) with rip := t5.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht5len
  · rw [ht6]; exact ht5k

/-- The length dispatch of the code loop, as a relation between
an opcode byte and its instruction length.  Mirrors the compare
chain at `0x330` and `decInstr` in `Sporos.Bytecode`. -/
def OpLen (op l : Nat) : Prop :=
  (op < 57 ∧ l = 1) ∨ (op = 128 ∧ l = 9) ∨
  ((op = 131 ∨ op = 132 ∨ op = 133 ∨ op = 140) ∧ l = 1) ∨
  ((129 ≤ op ∧ op ≤ 139 ∧ op ≠ 131 ∧ op ≠ 132 ∧ op ≠ 133) ∧ l = 5)

/-- Length dispatch, primitive opcodes: `cmp 57` not above-or-equal
falls to the length-1 assignment. -/
theorem parser_disp_lt57 {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hop : op < 57) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 4 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 1 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have h57W : 57 % W = 57 := Nat.mod_eq_of_lt (by unfold W; decide)
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  -- 0x330: cmp rax, 57
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub op (57 % W)).2, rip := s.rip + 4 } :=
    step_cmpI (by rw [hrip]; exact hc.p330) hr0
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .sub op (57 % W)).2, rip := s.rip + 4 } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x334 := by
    rw [ht1]; show s.rip + 4 = base + 0x334; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 3 = some false := by
    have hcf : t1.fl.cf = some true := by
      rw [ht1]
      show (aluF .sub op (57 % W)).2.cf = some true
      rw [h57W]; exact aluF_sub_cf_true hop
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht1fr : ∀ i, t1.reg i = s.reg i := by
    intro i
    rw [ht1]
    show s.reg i = s.reg i
    exact rfl
  have ht1mem : t1.mem = s.mem := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by rw [ht1]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]
  -- 0x334: jae not taken
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p334) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x336 := by
    rw [ht2]; show t1.rip + 2 = base + 0x336; rw [ht1rip]
  have ht2fr : ∀ i, t2.reg i = s.reg i := by
    intro i
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht1fr i
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).mem = s.mem
    rw [mem_with_rip]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1 with rip := t1.rip + 2 }).regs.length = 16
    rw [len_with_rip]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  -- 0x336: jmp 0x3ad
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with rip := base + 0x3ad } :=
    step_jmp (by rw [ht2rip]; exact hc.p336)
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with rip := base + 0x3ad } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x3ad := by rw [ht3]
  have ht3fr : ∀ i, t3.reg i = s.reg i := by
    intro i
    rw [ht3]
    show ({ t2 with rip := base + 0x3ad }).reg i = s.reg i
    rw [reg_with_rip]; exact ht2fr i
  have ht3mem : t3.mem = s.mem := by
    rw [ht3]
    show ({ t2 with rip := base + 0x3ad }).mem = s.mem
    rw [mem_with_rip]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2 with rip := base + 0x3ad }).regs.length = 16
    rw [len_with_rip]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x3ad: mov r10d, 1
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 10 (some (1 % W)) with rip := t3.rip + 6 } :=
    step_movI (by rw [ht3rip]; exact hc.p3ad)
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 10 (some (1 % W)) with rip := t3.rip + 6 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have hrun : run stage1Im 4 (Out.ok s) = Out.ok t4 := by
    rw [show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r1,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r2,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r3, r4]
  refine ⟨t4, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht4]; show t3.rip + 6 = base + 0x3b3; rw [ht3rip]
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).reg 10 =
      some 1
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide), h1W]
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).reg 0 =
      some op
    rw [reg_with_rip, reg_set_other (by decide : (0 : Nat) ≠ 10),
      ht3fr 0]; exact hr0
  · intro i hi
    rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht3fr i
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with rip := t3.rip + 6 }).mem =
      s.mem
    rw [mem_with_rip, mem_set]; exact ht3mem
  · rw [ht4]
    show ({ t3.set 10 (some (1 % W)) with
      rip := t3.rip + 6 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  · rw [ht4]; exact ht3k

/-- A subtraction of distinct in-range values is nonzero. -/
theorem sub_fst_ne_zero {op k : Nat} (hop : op < W) (hk : k < W)
    (hne : op ≠ k) : (aluF .sub op k).1 ≠ 0 := by
  rw [aluF_sub_fst]
  by_cases h : k ≤ op
  · rw [show op + W - k = (op - k) + W by omega, Nat.add_mod_right,
      Nat.mod_eq_of_lt (by omega)]
    omega
  · rw [Nat.mod_eq_of_lt (by omega)]
    omega

/-- A self-subtraction is zero. -/
theorem sub_fst_self (k : Nat) : (aluF .sub k k).1 = 0 := by
  rw [aluF_sub_fst, Nat.add_sub_cancel_left, Nat.mod_self]

/-- One unconditional jump step with full frame. -/
theorem jmp_step {a t len : Nat} {s : X86Int.St}
    (hdec : stage1Im a = some (.jmp t, len)) (hrip : s.rip = a) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧ s'.rip = t ∧
      (∀ i, s'.reg i = s.reg i) ∧ s'.mem = s.mem ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have e1 : X86Int.step stage1Im s = .ok { s with rip := t } :=
    step_jmp (by rw [hrip]; exact hdec)
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u = { s with rip := t } :=
    ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  refine ⟨t1, r1, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]
  · intro i
    rw [ht1]
    show ({ s with rip := t }).reg i = s.reg i
    rw [reg_with_rip]
  · rw [ht1]
  · rw [ht1]
  · rw [ht1]

/-- One `movI` step with full frame. -/
theorem movI_step {a d imm len : Nat} {s : X86Int.St}
    (hdec : stage1Im a = some (.movI d imm, len)) (hrip : s.rip = a)
    (hd : d < s.regs.length) :
    ∃ s' : X86Int.St,
      run stage1Im 1 (Out.ok s) = Out.ok s' ∧
      s'.rip = a + len ∧ s'.reg d = some (imm % W) ∧
      (∀ i, i ≠ d → s'.reg i = s.reg i) ∧ s'.mem = s.mem ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have e1 : X86Int.step stage1Im s =
      .ok { s.set d (some (imm % W)) with rip := s.rip + len } :=
    step_movI (by rw [hrip]; exact hdec)
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u =
      { s.set d (some (imm % W)) with rip := s.rip + len } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  refine ⟨t1, r1, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht1]; show s.rip + len = a + len; rw [hrip]
  · rw [ht1]
    show ({ s.set d (some (imm % W)) with rip := s.rip + len }).reg d =
      some (imm % W)
    rw [reg_with_rip, reg_set_self hd]
  · intro i hi
    rw [ht1]
    show ({ s.set d (some (imm % W)) with rip := s.rip + len }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi]
  · rw [ht1]
    show ({ s.set d (some (imm % W)) with rip := s.rip + len }).mem =
      s.mem
    rw [mem_with_rip, mem_set]
  · rw [ht1]
    show ({ s.set d (some (imm % W)) with
      rip := s.rip + len }).regs.length = s.regs.length
    rw [len_with_rip, len_set]
  · rw [ht1]; exact rfl

/-- One `cmpI`+`jcc` pair with a given condition outcome. -/
theorem cmpI_jcc_pair {a t k op clen cc : Nat} {b : Bool}
    {s : X86Int.St}
    (hcmp : stage1Im a = some (.cmpI 0 k, clen))
    (hjcc : stage1Im (a + clen) = some (.jcc cc t, 2))
    (hrip : s.rip = a) (hr0 : s.reg 0 = some op)
    (hfl : X86Int.cond (aluF .sub op (k % W)).2 cc = some b) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = (if b then t else a + clen + 2) ∧
      (∀ i, s'.reg i = s.reg i) ∧ s'.mem = s.mem ∧
      s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub op (k % W)).2, rip := s.rip + clen } :=
    step_cmpI (by rw [hrip]; exact hcmp) hr0
  obtain ⟨t1, ht1⟩ : ∃ u : X86Int.St, u =
      { s with fl := (aluF .sub op (k % W)).2, rip := s.rip + clen } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = a + clen := by
    rw [ht1]; show s.rip + clen = a + clen; rw [hrip]
  have ht1fl : X86Int.cond t1.fl cc = some b := by
    rw [ht1]
    show X86Int.cond (aluF .sub op (k % W)).2 cc = some b
    exact hfl
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := if b then t else t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hjcc) ht1fl
    exact h
  obtain ⟨t2, ht2⟩ : ∃ u : X86Int.St, u =
      { t1 with rip := if b then t else t1.rip + 2 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]
    show (if b then t else t1.rip + 2) = if b then t else a + clen + 2
    rw [ht1rip]
  · intro i
    rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).reg i =
      s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).mem = s.mem
    rw [mem_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).regs.length =
      s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := if b then t else t1.rip + 2 }).k = s.k
    rw [ht1]

/-- Condition outcome for `cmp op, k` with `op ≠ k` under cc5. -/
theorem fl_jne {op k : Nat} (hop : op < W) (hk : k < W)
    (hne : op ≠ k) :
    X86Int.cond (aluF .sub op (k % W)).2 5 = some true := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hzf : (aluF .sub op (k % W)).2.zf = some false := by
    rw [aluF_zf, hkmod]
    have hbeq : ((aluF .sub op k).1 == 0) = false := by
      cases h : ((aluF .sub op k).1 == 0) with
      | false => rfl
      | true =>
        rw [beq_iff_eq] at h
        exact absurd h (sub_fst_ne_zero hop hk hne)
    rw [hbeq]
  have hcc := cond_cc5 hzf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `op = k` under cc5. -/
theorem fl_je {op k : Nat} (hk : k < W) (heq : op = k) :
    X86Int.cond (aluF .sub op (k % W)).2 5 = some false := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hzf : (aluF .sub op (k % W)).2.zf = some true := by
    rw [aluF_zf, hkmod, heq]
    have hbeq : ((aluF .sub k k).1 == 0) = true := by
      rw [beq_iff_eq]; exact sub_fst_self k
    rw [hbeq]
  have hcc := cond_cc5 hzf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `k ≤ op` under cc3. -/
theorem fl_jae {op k : Nat} (hk : k < W) (h : ¬ op < k) :
    X86Int.cond (aluF .sub op (k % W)).2 3 = some true := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hcf : (aluF .sub op (k % W)).2.cf = some false := by
    rw [hkmod]; exact aluF_sub_cf h
  have hcc := cond_cc3 hcf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `op < k` under cc3. -/
theorem fl_jb {op k : Nat} (hk : k < W) (h : op < k) :
    X86Int.cond (aluF .sub op (k % W)).2 3 = some false := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  have hcf : (aluF .sub op (k % W)).2.cf = some true := by
    rw [hkmod]; exact aluF_sub_cf_true h
  have hcc := cond_cc3 hcf
  simpa using hcc

/-- Condition outcome for `cmp op, k` with `op ≤ k` under cc6. -/
theorem fl_jbe {op k : Nat} (hop : op < W) (hk : k < W)
    (h : op ≤ k) :
    X86Int.cond (aluF .sub op (k % W)).2 6 = some true := by
  have hkmod : k % W = k := Nat.mod_eq_of_lt hk
  by_cases hlt : op < k
  · have hcf : (aluF .sub op (k % W)).2.cf = some true := by
      rw [hkmod]; exact aluF_sub_cf_true hlt
    have hzf : (aluF .sub op (k % W)).2.zf =
        some ((aluF .sub op (k % W)).1 == 0) :=
      aluF_zf .sub _ _
    simp [X86Int.cond, hcf, hzf]
  · have heq : op = k := by omega
    have hcf : (aluF .sub op (k % W)).2.cf = some false := by
      rw [hkmod]; exact aluF_sub_cf (by omega)
    have hzf : (aluF .sub op (k % W)).2.zf = some true := by
      rw [aluF_zf, hkmod, heq]
      have hbeq : ((aluF .sub k k).1 == 0) = true := by
        rw [beq_iff_eq]; exact sub_fst_self k
      rw [hbeq]
    simp [X86Int.cond, hcf, hzf]

/-- Length dispatch, literal opcode `0x80`: length 9. -/
theorem parser_disp_128 {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (heq : op = 128) (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 7 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 9 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1rip' : s1.rip = base + 0x33b := hs1rip
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip' hs1r0
      (fl_je (by decide) heq)
  have hs2rip' : s2.rip = base + 0x344 := hs2rip
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    jmp_step hc.p344 hs2rip'
  have hs3len16 : s3.regs.length = 16 := by
    rw [hs3len, hs2len, hs1len]; exact hlen16
  obtain ⟨s4, hrun4, hs4rip, hs4r10, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    movI_step hc.p3a5 hs3rip (by rw [hs3len16]; decide)
  have hs4rip' : s4.rip = base + 0x3ab := hs4rip
  obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
    jmp_step hc.p3ab hs4rip'
  have hrun : run stage1Im 7 (Out.ok s) = Out.ok s5 := by
    rw [show (7 : Nat) = 2 + 5 by omega, run_add stage1Im 2 5, hrun1,
      show (5 : Nat) = 2 + 3 by omega, run_add stage1Im 2 3, hrun2,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, hrun3,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun4,
      hrun5]
  have h9W : 9 % W = 9 := Nat.mod_eq_of_lt (by unfold W; decide)
  refine ⟨s5, hrun, hs5rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs5fr 10, hs4r10, h9W]
  · rw [hs5fr 0, hs4fr 0 (by decide), hs3fr 0, hs2fr 0, hs1fr 0]
    exact hr0
  · intro i hi
    rw [hs5fr i, hs4fr i hi, hs3fr i, hs2fr i, hs1fr i]
  · rw [hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
  · rw [hs5len, hs4len]; exact hs3len16
  · rw [hs5k, hs4k, hs3k, hs2k, hs1k]

/-- Length dispatch, 5-byte opcodes (`129..139` except the
1-byte specials): the whole compare chain, ending at the
length-5 assignment. -/
theorem parser_disp_5 {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hlo : 129 ≤ op) (hhi : op ≤ 139)
    (h131 : op ≠ 131) (h132 : op ≠ 132) (h133 : op ≠ 133)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 18 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 5 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hopW : op < W := Nat.lt_of_le_of_lt hhi (by decide)
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1rip' : s1.rip = base + 0x33b := hs1rip
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip' hs1r0
      (fl_jne hopW (by decide) (by omega))
  have hs2rip' : s2.rip = base + 0x349 := hs2rip
  have hs2r0 : s2.reg 0 = some op := by rw [hs2fr 0]; exact hs1r0
  obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
    cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
      (fl_jne hopW (by decide) h131)
  have hs3rip' : s3.rip = base + 0x357 := hs3rip
  have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
      (fl_jne hopW (by decide) h132)
  have hs4rip' : s4.rip = base + 0x365 := hs4rip
  have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
  obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
    cmpI_jcc_pair hc.p365 hc.p36c hs4rip' hs4r0
      (fl_jne hopW (by decide) h133)
  have hs5rip' : s5.rip = base + 0x373 := hs5rip
  have hs5r0 : s5.reg 0 = some op := by rw [hs5fr 0]; exact hs4r0
  obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
    cmpI_jcc_pair hc.p373 hc.p37a hs5rip' hs5r0
      (fl_jne hopW (by decide) (by omega))
  have hs6rip' : s6.rip = base + 0x381 := hs6rip
  have hs6r0 : s6.reg 0 = some op := by rw [hs6fr 0]; exact hs5r0
  obtain ⟨s7, hrun7, hs7rip, hs7fr, hs7mem, hs7len, hs7k⟩ :=
    cmpI_jcc_pair hc.p381 hc.p388 hs6rip' hs6r0
      (fl_jae (by decide) (by omega))
  have hs7rip' : s7.rip = base + 0x38f := hs7rip
  have hs7r0 : s7.reg 0 = some op := by rw [hs7fr 0]; exact hs6r0
  obtain ⟨s8, hrun8, hs8rip, hs8fr, hs8mem, hs8len, hs8k⟩ :=
    cmpI_jcc_pair hc.p38f hc.p396 hs7rip' hs7r0
      (fl_jbe hopW (by decide) hhi)
  have hs8rip' : s8.rip = base + 0x39d := hs8rip
  have hs8len16 : s8.regs.length = 16 := by
    rw [hs8len, hs7len, hs6len, hs5len, hs4len, hs3len, hs2len,
      hs1len]; exact hlen16
  obtain ⟨s9, hrun9, hs9rip, hs9r10, hs9fr, hs9mem, hs9len, hs9k⟩ :=
    movI_step hc.p39d hs8rip' (by rw [hs8len16]; decide)
  have hs9rip' : s9.rip = base + 0x3a3 := hs9rip
  obtain ⟨s10, hrun10, hs10rip, hs10fr, hs10mem, hs10len, hs10k⟩ :=
    jmp_step hc.p3a3 hs9rip'
  have hrun : run stage1Im 18 (Out.ok s) = Out.ok s10 := by
    rw [show (18 : Nat) = 2 + 16 by omega, run_add stage1Im 2 16, hrun1,
      show (16 : Nat) = 2 + 14 by omega, run_add stage1Im 2 14, hrun2,
      show (14 : Nat) = 2 + 12 by omega, run_add stage1Im 2 12, hrun3,
      show (12 : Nat) = 2 + 10 by omega, run_add stage1Im 2 10, hrun4,
      show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8, hrun5,
      show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6, hrun6,
      show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4, hrun7,
      show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2, hrun8,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun9,
      hrun10]
  have h5W : 5 % W = 5 := Nat.mod_eq_of_lt (by unfold W; decide)
  refine ⟨s10, hrun, hs10rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [hs10fr 10, hs9r10, h5W]
  · rw [hs10fr 0, hs9fr 0 (by decide), hs8fr 0, hs7fr 0, hs6fr 0,
      hs5fr 0, hs4fr 0, hs3fr 0, hs2fr 0, hs1fr 0]
    exact hr0
  · intro i hi
    rw [hs10fr i, hs9fr i hi, hs8fr i, hs7fr i, hs6fr i, hs5fr i,
      hs4fr i, hs3fr i, hs2fr i, hs1fr i]
  · rw [hs10mem, hs9mem, hs8mem, hs7mem, hs6mem, hs5mem, hs4mem,
      hs3mem, hs2mem, hs1mem]
  · rw [hs10len, hs9len]; exact hs8len16
  · rw [hs10k, hs9k, hs8k, hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]

/-- Length dispatch, 1-byte special opcodes `131, 132, 133, 140`:
the chain falls out at the matching equality test to the
length-1 assignment. -/
theorem parser_disp_spec {op : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hspec : op = 131 ∨ op = 132 ∨ op = 133 ∨ op = 140)
    (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some 1 ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hopW : op < W := by
    have h1 : op ≤ 140 := by omega
    exact Nat.lt_of_le_of_lt h1 (by decide)
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    cmpI_jcc_pair hc.p330 hc.p334 hrip hr0
      (fl_jae (by decide) (by omega))
  have hs1rip' : s1.rip = base + 0x33b := hs1rip
  have hs1r0 : s1.reg 0 = some op := by rw [hs1fr 0]; exact hr0
  obtain ⟨s2, hrun2, hs2rip, hs2fr, hs2mem, hs2len, hs2k⟩ :=
    cmpI_jcc_pair hc.p33b hc.p342 hs1rip' hs1r0
      (fl_jne hopW (by decide) (by omega))
  have hs2rip' : s2.rip = base + 0x349 := hs2rip
  have hs2r0 : s2.reg 0 = some op := by rw [hs2fr 0]; exact hs1r0
  rcases hspec with h131 | h132 | h133 | h140
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_je (by decide) h131)
    have hs3rip' : s3.rip = base + 0x352 := hs3rip
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      jmp_step hc.p352 hs3rip'
    have hs4len16 : s4.regs.length = 16 := by
      rw [hs4len, hs3len, hs2len, hs1len]; exact hlen16
    obtain ⟨s5, hrun5, hs5rip, hs5r10, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      movI_step hc.p3ad hs4rip (by rw [hs4len16]; decide)
    have hrun : run stage1Im 8 (Out.ok s) = Out.ok s5 := by
      rw [show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6, hrun1,
        show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4, hrun2,
        show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2, hrun3,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun4,
        hrun5]
    refine ⟨8, s5, hrun, hs5rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs5r10, h1W]
    · rw [hs5fr 0 (by decide), hs4fr 0, hs3fr 0, hs2fr 0, hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs5fr i hi, hs4fr i, hs3fr i, hs2fr i, hs1fr i]
    · rw [hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
    · rw [hs5len]; exact hs4len16
    · rw [hs5k, hs4k, hs3k, hs2k, hs1k]
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_jne hopW (by decide) (by omega))
    have hs3rip' : s3.rip = base + 0x357 := hs3rip
    have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
        (fl_je (by decide) h132)
    have hs4rip' : s4.rip = base + 0x360 := hs4rip
    obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      jmp_step hc.p360 hs4rip'
    have hs5len16 : s5.regs.length = 16 := by
      rw [hs5len, hs4len, hs3len, hs2len, hs1len]; exact hlen16
    obtain ⟨s6, hrun6, hs6rip, hs6r10, hs6fr, hs6mem, hs6len, hs6k⟩ :=
      movI_step hc.p3ad hs5rip (by rw [hs5len16]; decide)
    have hrun : run stage1Im 10 (Out.ok s) = Out.ok s6 := by
      rw [show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8, hrun1,
        show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6, hrun2,
        show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4, hrun3,
        show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2, hrun4,
        show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, hrun5,
        hrun6]
    refine ⟨10, s6, hrun, hs6rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs6r10, h1W]
    · rw [hs6fr 0 (by decide), hs5fr 0, hs4fr 0, hs3fr 0, hs2fr 0,
        hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs6fr i hi, hs5fr i, hs4fr i, hs3fr i, hs2fr i, hs1fr i]
    · rw [hs6mem, hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
    · rw [hs6len]; exact hs5len16
    · rw [hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_jne hopW (by decide) (by omega))
    have hs3rip' : s3.rip = base + 0x357 := hs3rip
    have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
        (fl_jne hopW (by decide) (by omega))
    have hs4rip' : s4.rip = base + 0x365 := hs4rip
    have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
    obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      cmpI_jcc_pair hc.p365 hc.p36c hs4rip' hs4r0
        (fl_je (by decide) h133)
    have hs5rip' : s5.rip = base + 0x36e := hs5rip
    obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
      jmp_step hc.p36e hs5rip'
    have hs6len16 : s6.regs.length = 16 := by
      rw [hs6len, hs5len, hs4len, hs3len, hs2len, hs1len]
      exact hlen16
    obtain ⟨s7, hrun7, hs7rip, hs7r10, hs7fr, hs7mem, hs7len, hs7k⟩ :=
      movI_step hc.p3ad hs6rip (by rw [hs6len16]; decide)
    have hrun : run stage1Im 12 (Out.ok s) = Out.ok s7 := by
      rw [show (12 : Nat) = 2 + 10 by omega, run_add stage1Im 2 10,
        hrun1, show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8,
        hrun2, show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6,
        hrun3, show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4,
        hrun4, show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2,
        hrun5, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        hrun6, hrun7]
    refine ⟨12, s7, hrun, hs7rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs7r10, h1W]
    · rw [hs7fr 0 (by decide), hs6fr 0, hs5fr 0, hs4fr 0, hs3fr 0,
        hs2fr 0, hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs7fr i hi, hs6fr i, hs5fr i, hs4fr i, hs3fr i, hs2fr i,
        hs1fr i]
    · rw [hs7mem, hs6mem, hs5mem, hs4mem, hs3mem, hs2mem, hs1mem]
    · rw [hs7len]; exact hs6len16
    · rw [hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]
  · obtain ⟨s3, hrun3, hs3rip, hs3fr, hs3mem, hs3len, hs3k⟩ :=
      cmpI_jcc_pair hc.p349 hc.p350 hs2rip' hs2r0
        (fl_jne hopW (by decide) (by omega))
    have hs3rip' : s3.rip = base + 0x357 := hs3rip
    have hs3r0 : s3.reg 0 = some op := by rw [hs3fr 0]; exact hs2r0
    obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
      cmpI_jcc_pair hc.p357 hc.p35e hs3rip' hs3r0
        (fl_jne hopW (by decide) (by omega))
    have hs4rip' : s4.rip = base + 0x365 := hs4rip
    have hs4r0 : s4.reg 0 = some op := by rw [hs4fr 0]; exact hs3r0
    obtain ⟨s5, hrun5, hs5rip, hs5fr, hs5mem, hs5len, hs5k⟩ :=
      cmpI_jcc_pair hc.p365 hc.p36c hs4rip' hs4r0
        (fl_jne hopW (by decide) (by omega))
    have hs5rip' : s5.rip = base + 0x373 := hs5rip
    have hs5r0 : s5.reg 0 = some op := by rw [hs5fr 0]; exact hs4r0
    obtain ⟨s6, hrun6, hs6rip, hs6fr, hs6mem, hs6len, hs6k⟩ :=
      cmpI_jcc_pair hc.p373 hc.p37a hs5rip' hs5r0
        (fl_je (by decide) h140)
    have hs6rip' : s6.rip = base + 0x37c := hs6rip
    obtain ⟨s7, hrun7, hs7rip, hs7fr, hs7mem, hs7len, hs7k⟩ :=
      jmp_step hc.p37c hs6rip'
    have hs7len16 : s7.regs.length = 16 := by
      rw [hs7len, hs6len, hs5len, hs4len, hs3len, hs2len, hs1len]
      exact hlen16
    obtain ⟨s8, hrun8, hs8rip, hs8r10, hs8fr, hs8mem, hs8len, hs8k⟩ :=
      movI_step hc.p3ad hs7rip (by rw [hs7len16]; decide)
    have hrun : run stage1Im 14 (Out.ok s) = Out.ok s8 := by
      rw [show (14 : Nat) = 2 + 12 by omega, run_add stage1Im 2 12,
        hrun1, show (12 : Nat) = 2 + 10 by omega, run_add stage1Im 2 10,
        hrun2, show (10 : Nat) = 2 + 8 by omega, run_add stage1Im 2 8,
        hrun3, show (8 : Nat) = 2 + 6 by omega, run_add stage1Im 2 6,
        hrun4, show (6 : Nat) = 2 + 4 by omega, run_add stage1Im 2 4,
        hrun5, show (4 : Nat) = 2 + 2 by omega, run_add stage1Im 2 2,
        hrun6, show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1,
        hrun7, hrun8]
    refine ⟨14, s8, hrun, hs8rip, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hs8r10, h1W]
    · rw [hs8fr 0 (by decide), hs7fr 0, hs6fr 0, hs5fr 0, hs4fr 0,
        hs3fr 0, hs2fr 0, hs1fr 0]
      exact hr0
    · intro i hi
      rw [hs8fr i hi, hs7fr i, hs6fr i, hs5fr i, hs4fr i, hs3fr i,
        hs2fr i, hs1fr i]
    · rw [hs8mem, hs7mem, hs6mem, hs5mem, hs4mem, hs3mem, hs2mem,
        hs1mem]
    · rw [hs8len]; exact hs7len16
    · rw [hs8k, hs7k, hs6k, hs5k, hs4k, hs3k, hs2k, hs1k]

/-- The whole length dispatch, by `OpLen` case. -/
theorem parser_dispatch {op l : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x330) (hr0 : s.reg 0 = some op)
    (hOp : OpLen op l) (hlen16 : s.regs.length = 16) :
    ∃ n s', run stage1Im n (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3b3 ∧ s'.reg 10 = some l ∧
      s'.reg 0 = some op ∧
      (∀ i, i ≠ 10 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  rcases hOp with ⟨h1, hl⟩ | ⟨h1, hl⟩ | ⟨h1, hl⟩ | ⟨h1, hl⟩
  · subst hl
    obtain ⟨s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_lt57 hrip hr0 h1 hlen16
    exact ⟨4, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩
  · subst hl
    obtain ⟨s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_128 hrip hr0 h1 hlen16
    exact ⟨7, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩
  · subst hl
    obtain ⟨n, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_spec hrip hr0 h1 hlen16
    exact ⟨n, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩
  · subst hl
    obtain ⟨hlo, hhi, h131, h132, h133⟩ := h1
    obtain ⟨s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩ :=
      parser_disp_5 hrip hr0 hlo hhi h131 h132 h133 hlen16
    exact ⟨18, s', hrun, hrip', hr10, hr0', hfr, hmem, hlen, hk⟩

/-- `rdN` outside a single-byte write is unchanged. -/
theorem rdN_wr_other {m m' : Mem} {a v x n : Nat}
    (h : m.wr a v = some m') (hd : a + 1 ≤ x ∨ x + n ≤ a) :
    m'.rdN x n = m.rdN x n := by
  have hN : m.wrN a [v] = some m' := by simp [Mem.wrN, h]
  exact rdN_wrN_other n [v] m a x hN (fun i hi j _ => by
    simp at hi
    omega)

/-- Code-loop tail: length check, store the length byte in the
scratch at `S + (c - B)`, advance the cursor, decrement the
count, and jump back to the loop head. -/
theorem parser_code_tail {c e B S l n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x3b3) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr10 : s.reg 10 = some l)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (hr1 : s.reg 1 = some (n + 1))
    (hce : c + l ≤ e) (hcW : c < W) (heW : e < W)
    (hBc : B ≤ c) (hSW : S + (c - B) < W)
    (hl256 : l < 256) (hnW : n < W)
    (hwr : (s.mem.wr (S + (c - B)) l).isSome)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 11 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 6 = some (c + l) ∧
      s'.reg 1 = some n ∧ s'.reg 10 = some l ∧
      s'.mem.rd (S + (c - B)) = some l ∧
      (∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, S + (c - B) + 1 ≤ x ∨ x + k ≤ S + (c - B) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  -- 0x3b3: mov rdx, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 2 (some e) with rip := s.rip + 3 } :=
    step_mov (by rw [hrip]; exact hc.p3b3) hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 2 (some e) with rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x3b6 := by
    rw [ht1]; show s.rip + 3 = base + 0x3b6; rw [hrip]
  have ht1r2 : t1.reg 2 = some e := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (hlen 2 (by decide))]
  have ht1fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t1.reg i = s.reg i := by
    intro i _ hi2 _
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr6
  have ht1r10 : t1.reg 10 = some l := by
    rw [ht1fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht1mem : t1.mem = s.mem := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  -- 0x3b6: sub rdx, rsi
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } :=
    step_alu_sub (by rw [ht1rip]; exact hc.p3b6) ht1r2 ht1r6
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t1.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht2rip : t2.rip = base + 0x3b9 := by
    rw [ht2]; show t1.rip + 3 = base + 0x3b9; rw [ht1rip]
  have ht2r2 : t2.reg 2 = some (e - c) := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht1len]; decide), hsub]
  have ht2fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t2.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht1fr i hi1 hi2 hi6
  have ht2r10 : t2.reg 10 = some l := by
    rw [ht2fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).mem = s.mem
    rw [mem_set]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  -- 0x3b9: cmp rdx, r10
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with fl := (aluF .sub (e - c) l).2, rip := t2.rip + 3 } :=
    step_cmp (by rw [ht2rip]; exact hc.p3b9) ht2r2 ht2r10
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with fl := (aluF .sub (e - c) l).2, rip := t2.rip + 3 } :=
    ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x3bc := by
    rw [ht3]; show t2.rip + 3 = base + 0x3bc; rw [ht2rip]
  have ht3fl : X86Int.cond t3.fl 3 = some true := by
    have hcf : t3.fl.cf = some false := by
      rw [ht3]
      show (aluF .sub (e - c) l).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht3fr : ∀ i, t3.reg i = t2.reg i := by
    intro i
    rw [ht3]
    show t2.reg i = t2.reg i
    exact rfl
  have ht3mem : t3.mem = s.mem := by rw [ht3]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by rw [ht3]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  -- 0x3bc: jae 0x3c3
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := base + 0x3c3 } := by
    have h := step_jcc (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p3bc) ht3fl
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := base + 0x3c3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x3c3 := by rw [ht4]
  have ht4fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t4.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).reg i = s.reg i
    rw [reg_with_rip, ht3fr i]; exact ht2fr i hi1 hi2 hi6
  have ht4r6 : t4.reg 6 = some c := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).reg 6 = some c
    rw [reg_with_rip, ht3fr 6]
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht1r6
  have ht4r3 : t4.reg 3 = some B := by
    rw [ht4fr 3 (by decide) (by decide) (by decide)]; exact hr3
  have ht4r11 : t4.reg 11 = some S := by
    rw [ht4fr 11 (by decide) (by decide) (by decide)]; exact hr11
  have ht4r10 : t4.reg 10 = some l := by
    rw [ht4fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).mem = s.mem
    rw [mem_with_rip]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).regs.length = 16
    rw [len_with_rip]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  -- 0x3c3: mov rdx, rsi
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 2 (some c) with rip := t4.rip + 3 } :=
    step_mov (by rw [ht4rip]; exact hc.p3c3) ht4r6
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 2 (some c) with rip := t4.rip + 3 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x3c6 := by
    rw [ht5]; show t4.rip + 3 = base + 0x3c6; rw [ht4rip]
  have ht5r2 : t5.reg 2 = some c := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg 2 = some c
    rw [reg_with_rip, reg_set_self (by rw [ht4len]; decide)]
  have ht5fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t5.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi2]
    exact ht4fr i hi1 hi2 hi6
  have ht5r3 : t5.reg 3 = some B := by
    rw [ht5fr 3 (by decide) (by decide) (by decide)]; exact hr3
  have ht5r11 : t5.reg 11 = some S := by
    rw [ht5fr 11 (by decide) (by decide) (by decide)]; exact hr11
  have ht5r10 : t5.reg 10 = some l := by
    rw [ht5fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht5r6 : t5.reg 6 = some c := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht4r6
  have ht5r1 : t5.reg 1 = some (n + 1) := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).reg 1 =
      some (n + 1)
    rw [reg_with_rip, reg_set_other (by decide)]
    rw [ht4]
    show ({ t3 with rip := base + 0x3c3 }).reg 1 = some (n + 1)
    rw [reg_with_rip, ht3fr 1]
    rw [ht2]
    show (t1.set 2 (some (aluF .sub e c).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide), ht1]
    show ({ s.set 2 (some e) with rip := s.rip + 3 }).reg 1 =
      some (n + 1)
    rw [reg_with_rip, reg_set_other (by decide)]; exact hr1
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4.set 2 (some c) with rip := t4.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  -- 0x3c6: sub rdx, rbx
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 2 (some (aluF .sub c B).1) with
        fl := (aluF .sub c B).2, rip := t5.rip + 3 } :=
    step_alu_sub (by rw [ht5rip]; exact hc.p3c6) ht5r2 ht5r3
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 2 (some (aluF .sub c B).1) with
        fl := (aluF .sub c B).2, rip := t5.rip + 3 } := ⟨_, rfl⟩
  have hsub2 : (aluF .sub c B).1 = c - B := by
    rw [aluF_sub_fst, show c + W - B = (c - B) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht6rip : t6.rip = base + 0x3c9 := by
    rw [ht6]; show t5.rip + 3 = base + 0x3c9; rw [ht5rip]
  have ht6r2 : t6.reg 2 = some (c - B) := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg 2 = some (c - B)
    rw [reg_set_self (by rw [ht5len]; decide), hsub2]
  have ht6fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t6.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht5fr i hi1 hi2 hi6
  have ht6r11 : t6.reg 11 = some S := by
    rw [ht6fr 11 (by decide) (by decide) (by decide)]; exact hr11
  have ht6r6 : t6.reg 6 = some c := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht5r6
  have ht6r1 : t6.reg 1 = some (n + 1) := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide)]; exact ht5r1
  have ht6r10 : t6.reg 10 = some l := by
    rw [ht6fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht6mem : t6.mem = s.mem := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).mem = s.mem
    rw [mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 2 (some (aluF .sub c B).1)).regs.length = 16
    rw [len_set]; exact ht5len
  have ht6k : t6.k = s.k := by rw [ht6]; exact ht5k
  -- 0x3c9: add rdx, r11
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 2 (some (aluF .add (c - B) S).1) with
        fl := (aluF .add (c - B) S).2, rip := t6.rip + 3 } :=
    step_alu_add (by rw [ht6rip]; exact hc.p3c9) ht6r2 ht6r11
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 2 (some (aluF .add (c - B) S).1) with
        fl := (aluF .add (c - B) S).2, rip := t6.rip + 3 } := ⟨_, rfl⟩
  have hadd : (aluF .add (c - B) S).1 = S + (c - B) := by
    rw [aluF_add_fst, show c - B + S = S + (c - B) by omega,
      Nat.mod_eq_of_lt hSW]
  have ht7rip : t7.rip = base + 0x3cc := by
    rw [ht7]; show t6.rip + 3 = base + 0x3cc; rw [ht6rip]
  have ht7r2 : t7.reg 2 = some (S + (c - B)) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg 2 =
      some (S + (c - B))
    rw [reg_set_self (by rw [ht6len]; decide), hadd]
  have ht7fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t7.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg i = s.reg i
    rw [reg_set_other hi2]; exact ht6fr i hi1 hi2 hi6
  have ht7r6 : t7.reg 6 = some c := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg 6 = some c
    rw [reg_set_other (by decide)]; exact ht6r6
  have ht7r1 : t7.reg 1 = some (n + 1) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide)]; exact ht6r1
  have ht7r10 : t7.reg 10 = some l := by
    rw [ht7fr 10 (by decide) (by decide) (by decide)]; exact hr10
  have ht7mem : t7.mem = s.mem := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).mem = s.mem
    rw [mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add (c - B) S).1)).regs.length = 16
    rw [len_set]; exact ht6len
  have ht7k : t7.k = s.k := by rw [ht7]; exact ht6k
  -- 0x3cc: st8 [rdx], r10b
  obtain ⟨m8, hm8⟩ : ∃ m, s.mem.wr (S + (c - B)) l = some m :=
    Option.isSome_iff_exists.mp hwr
  have hm8' : t7.mem.wr (ea (S + (c - B)) 0) (l % 256) = some m8 := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hSW, ht7mem,
      Nat.mod_eq_of_lt hl256]
    exact hm8
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7 with mem := m8, rip := t7.rip + 3 } :=
    step_st8 (by rw [ht7rip]; exact hc.p3cc) ht7r2 ht7r10 hm8'
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7 with mem := m8, rip := t7.rip + 3 } := ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x3cf := by
    rw [ht8]; show t7.rip + 3 = base + 0x3cf; rw [ht7rip]
  have ht8fr : ∀ i, t8.reg i = t7.reg i := by
    intro i
    rw [ht8]
    show t7.reg i = t7.reg i
    exact rfl
  have ht8mem : t8.mem = m8 := by rw [ht8]
  have ht8len : t8.regs.length = 16 := by rw [ht8]; exact ht7len
  have ht8k : t8.k = s.k := by rw [ht8]; exact ht7k
  -- 0x3cf: add rsi, r10
  have ht8r6 : t8.reg 6 = some c := by rw [ht8fr 6]; exact ht7r6
  have ht8r10 : t8.reg 10 = some l := by rw [ht8fr 10]; exact ht7r10
  have ht8r1 : t8.reg 1 = some (n + 1) := by rw [ht8fr 1]; exact ht7r1
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 6 (some (aluF .add c l).1) with
        fl := (aluF .add c l).2, rip := t8.rip + 3 } :=
    step_alu_add (by rw [ht8rip]; exact hc.p3cf) ht8r6 ht8r10
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 6 (some (aluF .add c l).1) with
        fl := (aluF .add c l).2, rip := t8.rip + 3 } := ⟨_, rfl⟩
  have hadd2 : (aluF .add c l).1 = c + l := by
    rw [aluF_add_fst, Nat.mod_eq_of_lt (by omega)]
  have ht9rip : t9.rip = base + 0x3d2 := by
    rw [ht9]; show t8.rip + 3 = base + 0x3d2; rw [ht8rip]
  have ht9r6 : t9.reg 6 = some (c + l) := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).reg 6 = some (c + l)
    rw [reg_set_self (by rw [ht8len]; decide), hadd2]
  have ht9r1 : t9.reg 1 = some (n + 1) := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).reg 1 = some (n + 1)
    rw [reg_set_other (by decide)]; exact ht8r1
  have ht9fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t9.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).reg i = s.reg i
    rw [reg_set_other hi6, ht8fr i]; exact ht7fr i hi1 hi2 hi6
  have ht9mem : t9.mem = m8 := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).mem = m8
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 6 (some (aluF .add c l).1)).regs.length = 16
    rw [len_set]; exact ht8len
  have ht9k : t9.k = s.k := by rw [ht9]; exact ht8k
  -- 0x3d2: sub rcx, 1
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1) with
        fl := (aluF .sub (n + 1) (1 % W)).2, rip := t9.rip + 4 } :=
    step_aluI_sub (by rw [ht9rip]; exact hc.p3d2) ht9r1
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1) with
        fl := (aluF .sub (n + 1) (1 % W)).2, rip := t9.rip + 4 } :=
    ⟨_, rfl⟩
  have hsub3 : (aluF .sub (n + 1) (1 % W)).1 = n := by
    rw [h1W, aluF_sub_fst, show n + 1 + W - 1 = n + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt hnW]
  have ht10rip : t10.rip = base + 0x3d6 := by
    rw [ht10]; show t9.rip + 4 = base + 0x3d6; rw [ht9rip]
  have ht10r1 : t10.reg 1 = some n := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).reg 1 = some n
    rw [reg_set_self (by rw [ht9len]; decide), hsub3]
  have ht10r6 : t10.reg 6 = some (c + l) := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).reg 6 =
      some (c + l)
    rw [reg_set_other (by decide)]; exact ht9r6
  have ht10fr : ∀ i, i ≠ 1 → i ≠ 2 → i ≠ 6 → t10.reg i = s.reg i := by
    intro i hi1 hi2 hi6
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).reg i = s.reg i
    rw [reg_set_other hi1]; exact ht9fr i hi1 hi2 hi6
  have ht10mem : t10.mem = m8 := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).mem = m8
    rw [mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show (t9.set 1 (some (aluF .sub (n + 1) (1 % W)).1)).regs.length = 16
    rw [len_set]; exact ht9len
  have ht10k : t10.k = s.k := by rw [ht10]; exact ht9k
  -- 0x3d6: jmp loop head
  obtain ⟨t11, hrun11, ht11rip, ht11fr, ht11mem, ht11len, ht11k⟩ :=
    jmp_step hc.p3d6 ht10rip
  -- compose
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  have hrun : run stage1Im 11 (Out.ok s) = Out.ok t11 := by
    rw [show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r1,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r2,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r3,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r4,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r5,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r6,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r7,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r8,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r9,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r10,
      hrun11]
  have ht11mem' : t11.mem = m8 := by rw [ht11mem]; exact ht10mem
  refine ⟨t11, hrun, ht11rip, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht11fr 6]; exact ht10r6
  · rw [ht11fr 1]; exact ht10r1
  · rw [ht11fr 10, ht10fr 10 (by decide) (by decide) (by decide)]
    exact hr10
  · rw [ht11mem']
    have h := rd_wr_same hm8
    rwa [Nat.mod_eq_of_lt hl256] at h
  · intro i hi1 hi2 hi6
    rw [ht11fr i]; exact ht10fr i hi1 hi2 hi6
  · rw [ht11len]; exact ht10len
  · rw [ht11k]; exact ht10k
  · rw [ht11mem']; exact wr_wshape hm8
  · intro x k hd
    rw [ht11mem']
    exact rdN_wr_other hm8 hd

/-- Every instruction encoding starts with an opcode byte whose
dispatch length is the encoding length. -/
theorem enc_head (x : Instr) :
    ∃ op rest, x.enc = op :: rest ∧ OpLen op x.enc.length := by
  cases x with
  | prim p =>
    exact ⟨p.code, [], rfl, Or.inl ⟨Prim_code_lt p, rfl⟩⟩
  | lit n =>
    exact ⟨0x80, leBytes n 8, rfl,
      Or.inr (Or.inl ⟨rfl, by simp [Instr.enc, leBytes]⟩)⟩
  | str i =>
    exact ⟨0x81, leBytes i 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | pick n =>
    exact ⟨0x82, leBytes n 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | tor =>
    exact ⟨0x83, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inl rfl, rfl⟩))⟩
  | fromr =>
    exact ⟨0x84, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inr (Or.inl rfl), rfl⟩))⟩
  | rfetch =>
    exact ⟨0x85, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inr (Or.inr (Or.inl rfl)), rfl⟩))⟩
  | locget i =>
    exact ⟨0x86, leBytes i 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | locset i =>
    exact ⟨0x87, leBytes i 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | jmp t =>
    exact ⟨0x88, leBytes t 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | jz t =>
    exact ⟨0x89, leBytes t 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | jnz t =>
    exact ⟨0x8a, leBytes t 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | call w =>
    exact ⟨0x8b, leBytes w 4, rfl,
      Or.inr (Or.inr (Or.inr
        ⟨⟨by decide, by decide, by decide, by decide, by decide⟩,
          by simp [Instr.enc, leBytes]⟩))⟩
  | ret =>
    exact ⟨0x8c, [], rfl,
      Or.inr (Or.inr (Or.inl ⟨Or.inr (Or.inr (Or.inr rfl)), rfl⟩))⟩

/-- Code-loop head with instructions remaining: enter the body. -/
theorem parser_code_head_pos {n : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some (n + 1))
    (hnW : n + 1 < W) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x315 ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and (n + 1) (n + 1)).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p30b) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and (n + 1) (n + 1)).2, rip := s.rip + 3 } :=
    ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x30e := by
    rw [ht1]; show s.rip + 3 = base + 0x30e; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some true := by
    have hzf : t1.fl.zf = some false := by
      rw [ht1]
      show (aluF .and (n + 1) (n + 1)).2.zf = some false
      rw [aluF_zf, aluF_and_fst_self]
      simp [Nat.mod_eq_of_lt hnW]
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x315 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p30e) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x315 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]
  · intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x315 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht2]
    show ({ t1 with rip := base + 0x315 }).mem = s.mem
    rw [mem_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x315 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht2]
    show ({ t1 with rip := base + 0x315 }).k = s.k
    rw [ht1]

/-- Code-loop head with no instructions remaining: jump to the
word tail. -/
theorem parser_code_head_zero {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some 0) :
    ∃ s' : X86Int.St,
      run stage1Im 3 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x3db ∧
      (∀ i, s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = s.regs.length ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } :=
    step_test (by rw [hrip]; exact hc.p30b) hr1 hr1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .and 0 0).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x30e := by
    rw [ht1]; show s.rip + 3 = base + 0x30e; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 5 = some false := by
    have hzf : t1.fl.zf = some true := by
      rw [ht1]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have hcc := cond_cc5 hzf
    simpa using hcc
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := t1.rip + 2 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p30e) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := t1.rip + 2 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x310 := by
    rw [ht2]; show t1.rip + 2 = base + 0x310; rw [ht1rip]
  obtain ⟨t3, hrun3, ht3rip, ht3fr, ht3mem, ht3len, ht3k⟩ :=
    jmp_step hc.p310 ht2rip
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 3 (Out.ok s) = Out.ok t3 := by
    rw [show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r1,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r2, hrun3]
  refine ⟨t3, hrun, ht3rip, ?_, ?_, ?_, ?_⟩
  · intro i
    rw [ht3fr i, ht2]
    show ({ t1 with rip := t1.rip + 2 }).reg i = s.reg i
    rw [reg_with_rip, ht1]
    show s.reg i = s.reg i
    exact rfl
  · rw [ht3mem, ht2]
    show ({ t1 with rip := t1.rip + 2 }).mem = s.mem
    rw [mem_with_rip, ht1]
  · rw [ht3len, ht2]
    show ({ t1 with rip := t1.rip + 2 }).regs.length = s.regs.length
    rw [len_with_rip, ht1]
  · rw [ht3k, ht2]
    show ({ t1 with rip := t1.rip + 2 }).k = s.k
    rw [ht1]

/-- The length-scratch layout the code loop writes: at
`S + (pos - B)`, the byte length of the instruction at `pos`. -/
def LenLay (m : Mem) (B S c : Nat) : List Instr → Prop
  | [] => True
  | x :: xs => m.rd (S + (c - B)) = some x.enc.length ∧
      LenLay m B S (c + x.enc.length) xs

/-- A single-byte write is a one-byte `wrN`. -/
theorem wr_eq_wrN (m : Mem) (a v : Nat) : m.wr a v = m.wrN a [v] := by
  cases h : m.wr a v <;> simp [Mem.wrN, h]

/-- `rd` transport along an `rdN _ 1` equality. -/
theorem rd_of_rdN1 {m m' : Mem} {a v : Nat}
    (h : m'.rdN a 1 = m.rdN a 1) (hm : m.rd a = some v) :
    m'.rd a = some v := by
  have h1 : m.rdN a 1 = some [v] := by simp [Mem.rdN, hm]
  have h2 : m'.rdN a (0 + 1) = some (v :: []) := by
    show m'.rdN a 1 = some [v]
    rw [h]; exact h1
  exact (rdN_cons h2).1

/-- The instruction count never exceeds the byte count. -/
theorem code_length_le_bytes : ∀ (code : List Instr),
    code.length ≤ codeBytes code
  | [] => by decide
  | x :: xs => by
    have ih := code_length_le_bytes xs
    obtain ⟨op, rest, henc, _⟩ := enc_head x
    have hpos : 0 < x.enc.length := by
      rw [henc]; exact Nat.zero_lt_succ _
    show (x :: xs).length ≤ codeBytes (x :: xs)
    simp only [List.length_cons, codeBytes]
    omega

/-- One iteration of the code loop over the head instruction. -/
theorem parser_code_step {c e B S n : Nat} {x : Instr} {s : X86Int.St}
    (hrip : s.rip = base + 0x30b) (hr1 : s.reg 1 = some (n + 1))
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (henc : s.mem.rdN c x.enc.length = some x.enc)
    (hce : c + x.enc.length ≤ e) (hcW : c < W) (heW : e < W)
    (hnW : n + 1 < W) (hBc : B ≤ c) (hSW : S + (c - B) < W)
    (hwr : (s.mem.wr (S + (c - B)) x.enc.length).isSome)
    (hlen16 : s.regs.length = 16) :
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 6 = some (c + x.enc.length) ∧
      s'.reg 1 = some n ∧
      s'.mem.rd (S + (c - B)) = some x.enc.length ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ y k, S + (c - B) + 1 ≤ y ∨ y + k ≤ S + (c - B) →
        s'.mem.rdN y k = s.mem.rdN y k) := by
  obtain ⟨op, rest, henc_eq, hOp⟩ := enc_head x
  have hlenx : x.enc.length = rest.length + 1 := by
    rw [henc_eq]; simp
  have hlen256 : x.enc.length < 256 := by
    rcases hOp with ⟨_, hl⟩ | ⟨_, hl⟩ | ⟨_, hl⟩ | ⟨_, hl⟩ <;>
      omega
  have hrd : s.mem.rd c = some op := by
    have h : s.mem.rdN c (rest.length + 1) = some (op :: rest) := by
      rw [← hlenx, ← henc_eq]; exact henc
    exact (rdN_cons h).1
  obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
    parser_code_head_pos hrip hr1 hnW
  have hs1r6 : s1.reg 6 = some c := by rw [hs1fr 6]; exact hr6
  have hs1r15 : s1.reg 15 = some e := by rw [hs1fr 15]; exact hr15
  have hs1len16 : s1.regs.length = 16 := by rw [hs1len]; exact hlen16
  have hs1rd : s1.mem.rd c = some op := by rw [hs1mem]; exact hrd
  obtain ⟨s2, hrun2, hs2rip, hs2r0, hs2r6, hs2fr, hs2mem, hs2len,
      hs2k⟩ :=
    parser_code_bounds hs1rip hs1r6 hs1r15 (by omega) hcW heW hs1rd
      hs1len16
  have hs2len16 : s2.regs.length = 16 := hs2len
  obtain ⟨nd, s3, hrun3, hs3rip, hs3r10, hs3r0, hs3fr, hs3mem, hs3len,
      hs3k⟩ :=
    parser_dispatch hs2rip hs2r0 hOp hs2len16
  have hs3mem' : s3.mem = s.mem := by rw [hs3mem, hs2mem, hs1mem]
  have hs3r6 : s3.reg 6 = some c := by
    rw [hs3fr 6 (by decide)]; exact hs2r6
  have hs3r15 : s3.reg 15 = some e := by
    rw [hs3fr 15 (by decide), hs2fr 15 (by decide) (by decide),
      hs1fr 15]
    exact hr15
  have hs3r3 : s3.reg 3 = some B := by
    rw [hs3fr 3 (by decide), hs2fr 3 (by decide) (by decide), hs1fr 3]
    exact hr3
  have hs3r11 : s3.reg 11 = some S := by
    rw [hs3fr 11 (by decide), hs2fr 11 (by decide) (by decide),
      hs1fr 11]
    exact hr11
  have hs3r1 : s3.reg 1 = some (n + 1) := by
    rw [hs3fr 1 (by decide), hs2fr 1 (by decide) (by decide), hs1fr 1]
    exact hr1
  have hs3wr : (s3.mem.wr (S + (c - B)) x.enc.length).isSome := by
    rw [hs3mem']; exact hwr
  obtain ⟨s4, hrun4, hs4rip, hs4r6, hs4r1, hs4r10, hs4rd, hs4fr,
      hs4len, hs4k, hs4sh, hs4memfr⟩ :=
    parser_code_tail hs3rip hs3r6 hs3r15 hs3r10 hs3r3 hs3r11 hs3r1
      hce hcW heW hBc hSW hlen256 (by omega) hs3wr hs3len
  have hrun : run stage1Im (2 + 6 + nd + 11) (Out.ok s) = Out.ok s4 := by
    rw [show 2 + 6 + nd + 11 = 2 + (6 + (nd + 11)) by omega,
      run_add stage1Im 2 _, hrun1, run_add stage1Im 6 _, hrun2,
      run_add stage1Im nd 11, hrun3, hrun4]
  refine ⟨2 + 6 + nd + 11, s4, hrun, hs4rip, hs4r6, hs4r1, hs4rd, ?_,
    hs4len, ?_, ?_, ?_⟩
  · intro i hi0 hi1 hi2 hi6 hi10
    rw [hs4fr i hi1 hi2 hi6, hs3fr i hi10,
      hs2fr i hi0 hi2, hs1fr i]
  · rw [hs4k, hs3k, hs2k, hs1k]
  · rwa [hs3mem'] at hs4sh
  · intro y k hd
    rw [hs4memfr y k hd, hs3mem']

/-- The whole code loop over one word's instruction list: from
the loop head back to the loop head with count zero, the cursor
past the code bytes, `CodeSeq` preserved and `LenLay` written. -/
theorem parser_code_loop : ∀ (code : List Instr) {c e B S : Nat}
    {s : X86Int.St},
    s.rip = base + 0x30b → s.reg 1 = some code.length →
    s.reg 6 = some c → s.reg 15 = some e →
    s.reg 3 = some B → s.reg 11 = some S →
    CodeSeq s.mem c code →
    c + codeBytes code ≤ e → c < W → e < W →
    B ≤ c → S + (c + codeBytes code - B) < W →
    (S + (c + codeBytes code - B) ≤ c ∨ c + codeBytes code ≤ S + (c - B)) →
    (∀ a v, S + (c - B) ≤ a → a < S + (c + codeBytes code - B) →
      (s.mem.wr a v).isSome) →
    s.regs.length = 16 →
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 1 = some 0 ∧
      s'.reg 6 = some (c + codeBytes code) ∧
      CodeSeq s'.mem c code ∧ LenLay s'.mem B S c code ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ y k, S + (c + codeBytes code - B) ≤ y ∨
        y + k ≤ S + (c - B) → s'.mem.rdN y k = s.mem.rdN y k) := by
  intro code
  induction code with
  | nil =>
    intro c e B S s hrip hr1 hr6 _ _ _ _ _ hcW _ _ _ _ _ hlen16
    refine ⟨0, s, rfl, hrip, hr1, ?_, trivial, trivial, ?_, hlen16,
      rfl, rfl, ?_⟩
    · show s.reg 6 = some (c + codeBytes [])
      rw [show codeBytes [] = 0 from rfl]
      exact hr6
    · intro i _ _ _ _ _; exact rfl
    · intro y k _; exact rfl
  | cons x xs ih =>
    intro c e B S s hrip hr1 hr6 hr15 hr3 hr11 hseq hce hcW heW hBc
      hSW hdisj hwr hlen16
    obtain ⟨henc, hseqx⟩ := hseq
    have hcb : codeBytes (x :: xs) = x.enc.length + codeBytes xs := rfl
    have hpos : 0 < x.enc.length := by
      obtain ⟨op, rest, heq, _⟩ := enc_head x
      rw [heq]; exact Nat.zero_lt_succ _
    have hlenW : (x :: xs).length < W := by
      have h1 := code_length_le_bytes (x :: xs)
      omega
    have hr1' : s.reg 1 = some (xs.length + 1) := hr1
    have hwr0 : (s.mem.wr (S + (c - B)) x.enc.length).isSome := by
      apply hwr
      · omega
      · have hpos : 0 < x.enc.length := by
          obtain ⟨op, rest, heq, _⟩ := enc_head x
          rw [heq]; exact Nat.zero_lt_succ _
        omega
    have hSW0 : S + (c - B) < W := by omega
    obtain ⟨k1, s1, hrun1, hs1rip, hs1r6, hs1r1, hs1rd, hs1fr, hs1len,
        hs1k, hs1sh, hs1memfr⟩ :=
      parser_code_step hrip hr1' hr6 hr15 hr3 hr11 henc (by omega)
        hcW heW hlenW hBc hSW0 hwr0 hlen16
    have hs1r15 : s1.reg 15 = some e := by
      rw [hs1fr 15 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hr15
    have hs1r3 : s1.reg 3 = some B := by
      rw [hs1fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hr3
    have hs1r11 : s1.reg 11 = some S := by
      rw [hs1fr 11 (by decide) (by decide) (by decide) (by decide)
        (by decide)]
      exact hr11
    have hs1seq : CodeSeq s1.mem (c + x.enc.length) xs := by
      apply CodeSeq_of_frame hseqx
      intro y k hy hk
      apply hs1memfr
      omega
    have hs1wr : ∀ a v, S + (c + x.enc.length - B) ≤ a →
        a < S + (c + x.enc.length + codeBytes xs - B) →
        (s1.mem.wr a v).isSome := by
      intro a v ha1 ha2
      have h1 : (s.mem.wrN a [v]).isSome := by
        rw [← wr_eq_wrN]
        apply hwr <;> omega
      have h2 := wrN_some_of_wshape _ s.mem s1.mem hs1sh _ h1
      rwa [← wr_eq_wrN] at h2
    obtain ⟨k2, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2seq, hs2lay, hs2fr,
        hs2len, hs2k, hs2sh, hs2memfr⟩ :=
      ih hs1rip hs1r1 hs1r6 hs1r15 hs1r3 hs1r11 hs1seq (by omega)
        (by omega) heW (by omega) (by omega) (by omega) hs1wr hs1len
    have hrun : run stage1Im (k1 + k2) (Out.ok s) = Out.ok s2 := by
      rw [run_add stage1Im k1 k2, hrun1, hrun2]
    have hseq' : CodeSeq s2.mem c (x :: xs) := by
      constructor
      · have h1 : s2.mem.rdN c x.enc.length =
            s1.mem.rdN c x.enc.length := by
          apply hs2memfr
          omega
        have h2 : s1.mem.rdN c x.enc.length =
            s.mem.rdN c x.enc.length := by
          apply hs1memfr
          omega
        rw [h1, h2]; exact henc
      · exact hs2seq
    have hlay' : LenLay s2.mem B S c (x :: xs) := by
      constructor
      · have h1 : s2.mem.rdN (S + (c - B)) 1 =
            s1.mem.rdN (S + (c - B)) 1 := by
          apply hs2memfr
          omega
        exact rd_of_rdN1 h1 hs1rd
      · exact hs2lay
    refine ⟨k1 + k2, s2, hrun, hs2rip, hs2r1, ?_, hseq', hlay', ?_,
      hs2len, ?_, ?_, ?_⟩
    · show s2.reg 6 = some (c + codeBytes (x :: xs))
      rw [hcb]
      have h := hs2r6
      rwa [show c + x.enc.length + codeBytes xs =
        c + (x.enc.length + codeBytes xs) by omega] at h
    · intro i hi0 hi1 hi2 hi6 hi10
      rw [hs2fr i hi0 hi1 hi2 hi6 hi10]
      exact hs1fr i hi0 hi1 hi2 hi6 hi10
    · rw [hs2k, hs1k]
    · rw [hs2sh, hs1sh]
    · intro y k hd
      have h1 : s2.mem.rdN y k = s1.mem.rdN y k := by
        apply hs2memfr
        omega
      have h2 : s1.mem.rdN y k = s.mem.rdN y k := by
        apply hs1memfr
        omega
      rw [h1, h2]

/-- Code-loop entry: push the outer word counter and load this
word's instruction count from slot 24. -/
theorem parser_code_enter {sp p m nc : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x306) (hr1 : s.reg 1 = some m)
    (hr7 : s.reg 7 = some p) (hr4 : s.reg 4 = some sp)
    (hroom : StackRoom s 1) (hsp8 : 8 ≤ sp) (hspW : sp < W)
    (hpW : p + 24 < W)
    (hslot : rd64 s.mem (p + 24) = some nc)
    (hdisj : p + 24 + 8 ≤ sp - 8 ∨ sp ≤ p + 24)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 2 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x30b ∧ s'.reg 1 = some nc ∧
      s'.reg 4 = some (sp - 8) ∧
      rd64 s'.mem (sp - 8) = some (m % W) ∧
      rd64 s'.mem (p + 24) = some nc ∧
      (∀ i, i ≠ 1 → i ≠ 4 → s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, x + k ≤ sp - 8 ∨ sp ≤ x →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hsp8mod : (sp + W - 8) % W = sp - 8 := by
    have h : sp + W - 8 = (sp - 8) + W := by omega
    rw [h, Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  obtain ⟨m1, hm1raw⟩ : ∃ mm, wr64 s.mem ((sp + W - 8 * (0 + 1)) % W) m =
      some mm :=
    Option.isSome_iff_exists.mp (hroom sp hr4 0 (by decide) m)
  have hm1 : wr64 s.mem ((sp + W - 8) % W) m = some m1 := hm1raw
  have hm1wr : wr64 s.mem (sp - 8) m = some m1 := by
    rwa [hsp8mod] at hm1
  -- 0x306: push rcx
  have e1 : X86Int.step stage1Im s =
      .ok { s.set 4 (some ((sp + W - 8) % W)) with
        mem := m1, rip := s.rip + 1 } :=
    step_push (by rw [hrip]; exact hc.p306) hr1 hr4 hm1
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s.set 4 (some ((sp + W - 8) % W)) with
        mem := m1, rip := s.rip + 1 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x307 := by
    rw [ht1]; show s.rip + 1 = base + 0x307; rw [hrip]
  have ht1r4 : t1.reg 4 = some (sp - 8) := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 4 = some (sp - 8)
    rw [reg_set_self (hlen 4 (by decide)), hsp8mod]
  have ht1r7 : t1.reg 7 = some p := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg 7 = some p
    rw [reg_set_other (by decide)]; exact hr7
  have ht1fr : ∀ i, i ≠ 4 → t1.reg i = s.reg i := by
    intro i hi
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).reg i = s.reg i
    rw [reg_set_other hi]
  have ht1mem : t1.mem = m1 := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show (s.set 4 (some ((sp + W - 8) % W))).regs.length = 16
    rw [len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  have ht1slot : rd64 t1.mem (p + 24) = some nc := by
    rw [ht1mem]
    have h := rd64_wr64_other hm1wr (m := s.mem) (x := p + 24) (by
      rcases hdisj with h | h
      · exact Or.inr h
      · exact Or.inl (by omega))
    rw [h]; exact hslot
  -- 0x307: ld rcx, [rdi+24]
  have hld : rd64 t1.mem (ea p 24) = some nc := by
    unfold ea
    rwa [Nat.mod_eq_of_lt hpW]
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1.set 1 (some nc) with rip := t1.rip + 4 } :=
    step_ld (by rw [ht1rip]; exact hc.p307) ht1r7 hld
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1.set 1 (some nc) with rip := t1.rip + 4 } := ⟨_, rfl⟩
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  have hrun : run stage1Im 2 (Out.ok s) = Out.ok t2 := by
    rw [show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r1, r2]
  have ht2mem : t2.mem = m1 := by
    rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).mem = m1
    rw [mem_with_rip, mem_set]; exact ht1mem
  refine ⟨t2, hrun, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht2]; show t1.rip + 4 = base + 0x30b; rw [ht1rip]
  · rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).reg 1 = some nc
    rw [reg_with_rip, reg_set_self (by rw [ht1len]; decide)]
  · rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).reg 4 =
      some (sp - 8)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht1r4
  · rw [ht2mem]; exact rd64_wr64 hm1wr
  · rw [ht2mem]
    have h := rd64_wr64_other hm1wr (m := s.mem) (x := p + 24) (by
      rcases hdisj with h | h
      · exact Or.inr h
      · exact Or.inl (by omega))
    rw [h]; exact hslot
  · intro i hi1 hi4
    rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi1]; exact ht1fr i hi4
  · rw [ht2]
    show ({ t1.set 1 (some nc) with rip := t1.rip + 4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht1len
  · rw [ht2]; exact ht1k
  · rw [ht2mem]; exact wrN_wshape _ _ _ hm1wr
  · intro x k hd
    rw [ht2mem]
    apply rdN_wr64_other hm1wr
    rcases hd with h | h
    · exact Or.inr h
    · exact Or.inl (by omega)

set_option maxHeartbeats 0 in
/-- Word tail after the code loop: pop the outer counter, store
the ann pointer at slot 48, skip `12 * nann` annotation bytes,
advance the entry pointer and decrement the counter, back to
the word-loop head. -/
theorem parser_word_tail {sp1 p c e r na : Nat} {m48 : Mem}
    {s : X86Int.St}
    (hrip : s.rip = base + 0x3db) (hr4 : s.reg 4 = some sp1)
    (hr6 : s.reg 6 = some c) (hr7 : s.reg 7 = some p)
    (hr15 : s.reg 15 = some e)
    (hpop : rd64 s.mem sp1 = some (r + 1))
    (hspW : sp1 + 8 < W) (hrW : r < W)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hnaW : 12 * na < W) (hce : c + 12 * na ≤ e)
    (hslot32 : rd64 s.mem (p + 32) = some na)
    (hm48 : wr64 s.mem (p + 48) c = some m48)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 18 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 6 = some (c + 12 * na) ∧
      s'.reg 7 = some (p + 64) ∧ s'.reg 1 = some r ∧
      s'.reg 4 = some (sp1 + 8) ∧
      rd64 s'.mem (p + 48) = some (c % W) ∧
      rd64 s'.mem (p + 32) = some na ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧
      (∀ x k, p + 48 + 8 ≤ x ∨ x + k ≤ p + 48 →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have h1W : 1 % W = 1 := Nat.mod_eq_of_lt (by unfold W; decide)
  have h64W : 64 % W = 64 := Nat.mod_eq_of_lt (by unfold W; decide)
  -- 0x3db: pop rcx
  have e1 : X86Int.step stage1Im s =
      .ok { (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
        rip := s.rip + 1 } :=
    step_pop (by rw [hrip]; exact hc.p3db) hr4 hpop
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
        rip := s.rip + 1 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x3dc := by
    rw [ht1]; show s.rip + 1 = base + 0x3dc; rw [hrip]
  have ht1r1 : t1.reg 1 = some (r + 1) := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 1 = some (r + 1)
    rw [reg_with_rip, reg_set_self (by
      show 1 < (s.set 4 (some ((sp1 + 8) % W))).regs.length
      rw [len_set]; omega)]
  have ht1r4 : t1.reg 4 = some (sp1 + 8) := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 4 = some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide : (4 : Nat) ≠ 1),
      reg_set_self (hlen 4 (by decide)), Nat.mod_eq_of_lt hspW]
  have ht1fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t1.reg i = s.reg i := by
    intro i _ hi1 _ hi4 _ _
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi1, reg_set_other hi4]
  have ht1r6 : t1.reg 6 = some c := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide : (6 : Nat) ≠ 1),
      reg_set_other (by decide : (6 : Nat) ≠ 4)]
    exact hr6
  have ht1r7 : t1.reg 7 = some p := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).reg 7 = some p
    rw [reg_with_rip, reg_set_other (by decide : (7 : Nat) ≠ 1),
      reg_set_other (by decide : (7 : Nat) ≠ 4)]
    exact hr7
  have ht1r15 : t1.reg 15 = some e := by
    rw [ht1fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hr15
  have ht1mem : t1.mem = s.mem := by rw [ht1]; rfl
  have ht1len : t1.regs.length = 16 := by
    rw [ht1]
    show ({ (s.set 4 (some ((sp1 + 8) % W))).set 1 (some (r + 1)) with
      rip := s.rip + 1 }).regs.length = 16
    rw [len_with_rip, len_set, len_set]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]; rfl
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  -- 0x3dc: st [rdi+48], rsi
  have hm48' : wr64 t1.mem (p + 48) c = some m48 := by
    rw [ht1mem]; exact hm48
  obtain ⟨t2, hrun2, ht2rip, ht2mem, ht2slot, ht2fr, ht2len, ht2k,
      ht2sh, ht2memfr⟩ :=
    store_slot (site := base + 0x3dc) (off := 48) (slen := 4) (r := 6)
      hc.p3dc ht1rip ht1r7 ht1r6 (by omega) hm48'
  have ht2rip' : t2.rip = base + 0x3e0 := ht2rip
  have ht2len16 : t2.regs.length = 16 := by
    rw [ht2len]; exact ht1len
  have ht2slot32 : rd64 t2.mem (p + 32) = some na := by
    rw [ht2mem]
    have h := rd64_wr64_other hm48 (m := s.mem) (x := p + 32)
      (Or.inr (by omega))
    rw [h]; exact hslot32
  -- 0x3e0: ld rax, [rdi+32]
  have ht2r7 : t2.reg 7 = some p := by rw [ht2fr 7]; exact ht1r7
  have hld : rd64 t2.mem (ea p 32) = some na := by
    unfold ea
    rwa [Nat.mod_eq_of_lt (by omega : p + 32 < W)]
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2.set 0 (some na) with rip := t2.rip + 4 } :=
    step_ld (by rw [ht2rip']; exact hc.p3e0) ht2r7 hld
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2.set 0 (some na) with rip := t2.rip + 4 } := ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x3e4 := by
    rw [ht3]; show t2.rip + 4 = base + 0x3e4; rw [ht2rip']
  have ht3r0 : t3.reg 0 = some na := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 0 = some na
    rw [reg_with_rip, reg_set_self (by rw [ht2len16]; decide)]
  have ht3fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 →
      t3.reg i = s.reg i := by
    intro i hi0 hi1 hi2 hi4 hi6 hi7
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi0, ht2fr i]
    exact ht1fr i hi0 hi1 hi2 hi4 hi6 hi7
  have ht3r6 : t3.reg 6 = some c := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 6 = some c
    rw [reg_with_rip, reg_set_other (by decide), ht2fr 6]
    exact ht1r6
  have ht3r7 : t3.reg 7 = some p := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 7 = some p
    rw [reg_with_rip, reg_set_other (by decide)]
    exact ht2r7
  have ht3r1 : t3.reg 1 = some (r + 1) := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 1 =
      some (r + 1)
    rw [reg_with_rip, reg_set_other (by decide), ht2fr 1]
    exact ht1r1
  have ht3r4 : t3.reg 4 = some (sp1 + 8) := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).reg 4 =
      some (sp1 + 8)
    rw [reg_with_rip, reg_set_other (by decide), ht2fr 4]
    exact ht1r4
  have ht3r15 : t3.reg 15 = some e := by
    rw [ht3fr 15 (by decide) (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hr15
  have ht3mem : t3.mem = m48 := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).mem = m48
    rw [mem_with_rip, mem_set]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by
    rw [ht3]
    show ({ t2.set 0 (some na) with rip := t2.rip + 4 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht2len16
  have ht3k : t3.k = s.k := by
    rw [ht3]
    show t2.k = s.k
    rw [ht2k]; exact ht1k
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  -- 0x3e4: mov rdx, rax
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3.set 2 (some na) with rip := t3.rip + 3 } :=
    step_mov (by rw [ht3rip]; exact hc.p3e4) ht3r0
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3.set 2 (some na) with rip := t3.rip + 3 } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x3e7 := by
    rw [ht4]; show t3.rip + 3 = base + 0x3e7; rw [ht3rip]
  have ht4r2 : t4.reg 2 = some na := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).reg 2 = some na
    rw [reg_with_rip, reg_set_self (by rw [ht3len]; decide)]
  have ht4r0 : t4.reg 0 = some na := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).reg 0 = some na
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht3r0
  have ht4fr : ∀ i, i ≠ 0 → i ≠ 2 → t4.reg i = t3.reg i := by
    intro i _ hi2
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).reg i =
      t3.reg i
    rw [reg_with_rip, reg_set_other hi2]
  have ht4mem : t4.mem = m48 := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).mem = m48
    rw [mem_with_rip, mem_set]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3.set 2 (some na) with rip := t3.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  -- 0x3e7: add rax, rax (2*na)
  have hdbl : ∀ v : Nat, 2 * v < W → (aluF .add v v).1 = 2 * v := by
    intro v hv
    rw [aluF_add_fst, show v + v = 2 * v by omega,
      Nat.mod_eq_of_lt hv]
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 0 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t4.rip + 3 } :=
    step_alu_add (by rw [ht4rip]; exact hc.p3e7) ht4r0 ht4r0
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 0 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t4.rip + 3 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x3ea := by
    rw [ht5]; show t4.rip + 3 = base + 0x3ea; rw [ht4rip]
  have ht5r0 : t5.reg 0 = some (2 * na) := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).reg 0 = some (2 * na)
    rw [reg_set_self (by rw [ht4len]; decide),
      hdbl na (by omega)]
  have ht5r2 : t5.reg 2 = some na := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).reg 2 = some na
    rw [reg_set_other (by decide)]; exact ht4r2
  have ht5fr : ∀ i, i ≠ 0 → i ≠ 2 → t5.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).reg i = t3.reg i
    rw [reg_set_other hi0]; exact ht4fr i hi0 hi2
  have ht5mem : t5.mem = m48 := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).mem = m48
    rw [mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show (t4.set 0 (some (aluF .add na na).1)).regs.length = 16
    rw [len_set]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  -- 0x3ea: add rax, rax (4*na)
  have e6 : X86Int.step stage1Im t5 =
      .ok { t5.set 0 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t5.rip + 3 } :=
    step_alu_add (by rw [ht5rip]; exact hc.p3ea) ht5r0 ht5r0
  obtain ⟨t6, ht6⟩ : ∃ t : X86Int.St, t =
      { t5.set 0 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t5.rip + 3 } :=
    ⟨_, rfl⟩
  have ht6rip : t6.rip = base + 0x3ed := by
    rw [ht6]; show t5.rip + 3 = base + 0x3ed; rw [ht5rip]
  have ht6r0 : t6.reg 0 = some (4 * na) := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).reg 0 =
      some (4 * na)
    rw [reg_set_self (by rw [ht5len]; decide),
      hdbl (2 * na) (by omega)]
    congr 1
    omega
  have ht6r2 : t6.reg 2 = some na := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).reg 2 =
      some na
    rw [reg_set_other (by decide)]; exact ht5r2
  have ht6fr : ∀ i, i ≠ 0 → i ≠ 2 → t6.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi0]; exact ht5fr i hi0 hi2
  have ht6mem : t6.mem = m48 := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).mem = m48
    rw [mem_set]; exact ht5mem
  have ht6len : t6.regs.length = 16 := by
    rw [ht6]
    show (t5.set 0 (some (aluF .add (2 * na) (2 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht5len
  have ht6k : t6.k = s.k := by rw [ht6]; exact ht5k
  have r6 : run stage1Im 1 (Out.ok t5) = Out.ok t6 := by
    rw [run_succ_ok stage1Im 0 t5, e6, ← ht6, run_zero]
  -- 0x3ed: add rdx, rdx (2*na)
  have e7 : X86Int.step stage1Im t6 =
      .ok { t6.set 2 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t6.rip + 3 } :=
    step_alu_add (by rw [ht6rip]; exact hc.p3ed) ht6r2 ht6r2
  obtain ⟨t7, ht7⟩ : ∃ t : X86Int.St, t =
      { t6.set 2 (some (aluF .add na na).1) with
        fl := (aluF .add na na).2, rip := t6.rip + 3 } := ⟨_, rfl⟩
  have ht7rip : t7.rip = base + 0x3f0 := by
    rw [ht7]; show t6.rip + 3 = base + 0x3f0; rw [ht6rip]
  have ht7r2 : t7.reg 2 = some (2 * na) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).reg 2 = some (2 * na)
    rw [reg_set_self (by rw [ht6len]; decide),
      hdbl na (by omega)]
  have ht7r0 : t7.reg 0 = some (4 * na) := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).reg 0 = some (4 * na)
    rw [reg_set_other (by decide)]; exact ht6r0
  have ht7fr : ∀ i, i ≠ 0 → i ≠ 2 → t7.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).reg i = t3.reg i
    rw [reg_set_other hi2]; exact ht6fr i hi0 hi2
  have ht7mem : t7.mem = m48 := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).mem = m48
    rw [mem_set]; exact ht6mem
  have ht7len : t7.regs.length = 16 := by
    rw [ht7]
    show (t6.set 2 (some (aluF .add na na).1)).regs.length = 16
    rw [len_set]; exact ht6len
  have ht7k : t7.k = s.k := by rw [ht7]; exact ht6k
  have r7 : run stage1Im 1 (Out.ok t6) = Out.ok t7 := by
    rw [run_succ_ok stage1Im 0 t6, e7, ← ht7, run_zero]
  -- 0x3f0: add rdx, rdx (4*na)
  have e8 : X86Int.step stage1Im t7 =
      .ok { t7.set 2 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t7.rip + 3 } :=
    step_alu_add (by rw [ht7rip]; exact hc.p3f0) ht7r2 ht7r2
  obtain ⟨t8, ht8⟩ : ∃ t : X86Int.St, t =
      { t7.set 2 (some (aluF .add (2 * na) (2 * na)).1) with
        fl := (aluF .add (2 * na) (2 * na)).2, rip := t7.rip + 3 } :=
    ⟨_, rfl⟩
  have ht8rip : t8.rip = base + 0x3f3 := by
    rw [ht8]; show t7.rip + 3 = base + 0x3f3; rw [ht7rip]
  have ht8r2 : t8.reg 2 = some (4 * na) := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).reg 2 =
      some (4 * na)
    rw [reg_set_self (by rw [ht7len]; decide),
      hdbl (2 * na) (by omega)]
    congr 1
    omega
  have ht8r0 : t8.reg 0 = some (4 * na) := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).reg 0 =
      some (4 * na)
    rw [reg_set_other (by decide)]; exact ht7r0
  have ht8fr : ∀ i, i ≠ 0 → i ≠ 2 → t8.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi2]; exact ht7fr i hi0 hi2
  have ht8mem : t8.mem = m48 := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).mem = m48
    rw [mem_set]; exact ht7mem
  have ht8len : t8.regs.length = 16 := by
    rw [ht8]
    show (t7.set 2 (some (aluF .add (2 * na) (2 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht7len
  have ht8k : t8.k = s.k := by rw [ht8]; exact ht7k
  have r8 : run stage1Im 1 (Out.ok t7) = Out.ok t8 := by
    rw [run_succ_ok stage1Im 0 t7, e8, ← ht8, run_zero]
  -- 0x3f3: add rdx, rdx (8*na)
  have e9 : X86Int.step stage1Im t8 =
      .ok { t8.set 2 (some (aluF .add (4 * na) (4 * na)).1) with
        fl := (aluF .add (4 * na) (4 * na)).2, rip := t8.rip + 3 } :=
    step_alu_add (by rw [ht8rip]; exact hc.p3f3) ht8r2 ht8r2
  obtain ⟨t9, ht9⟩ : ∃ t : X86Int.St, t =
      { t8.set 2 (some (aluF .add (4 * na) (4 * na)).1) with
        fl := (aluF .add (4 * na) (4 * na)).2, rip := t8.rip + 3 } :=
    ⟨_, rfl⟩
  have ht9rip : t9.rip = base + 0x3f6 := by
    rw [ht9]; show t8.rip + 3 = base + 0x3f6; rw [ht8rip]
  have ht9r2 : t9.reg 2 = some (8 * na) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).reg 2 =
      some (8 * na)
    rw [reg_set_self (by rw [ht8len]; decide),
      hdbl (4 * na) (by omega)]
    congr 1
    omega
  have ht9r0 : t9.reg 0 = some (4 * na) := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).reg 0 =
      some (4 * na)
    rw [reg_set_other (by decide)]; exact ht8r0
  have ht9fr : ∀ i, i ≠ 0 → i ≠ 2 → t9.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi2]; exact ht8fr i hi0 hi2
  have ht9mem : t9.mem = m48 := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).mem = m48
    rw [mem_set]; exact ht8mem
  have ht9len : t9.regs.length = 16 := by
    rw [ht9]
    show (t8.set 2 (some (aluF .add (4 * na) (4 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht8len
  have ht9k : t9.k = s.k := by rw [ht9]; exact ht8k
  have r9 : run stage1Im 1 (Out.ok t8) = Out.ok t9 := by
    rw [run_succ_ok stage1Im 0 t8, e9, ← ht9, run_zero]
  -- 0x3f6: add rax, rdx (12*na)
  have e10 : X86Int.step stage1Im t9 =
      .ok { t9.set 0 (some (aluF .add (4 * na) (8 * na)).1) with
        fl := (aluF .add (4 * na) (8 * na)).2, rip := t9.rip + 3 } :=
    step_alu_add (by rw [ht9rip]; exact hc.p3f6) ht9r0 ht9r2
  obtain ⟨t10, ht10⟩ : ∃ t : X86Int.St, t =
      { t9.set 0 (some (aluF .add (4 * na) (8 * na)).1) with
        fl := (aluF .add (4 * na) (8 * na)).2, rip := t9.rip + 3 } :=
    ⟨_, rfl⟩
  have ht10rip : t10.rip = base + 0x3f9 := by
    rw [ht10]; show t9.rip + 3 = base + 0x3f9; rw [ht9rip]
  have ht10r0 : t10.reg 0 = some (12 * na) := by
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).reg 0 =
      some (12 * na)
    rw [reg_set_self (by rw [ht9len]; decide), aluF_add_fst,
      show 4 * na + 8 * na = 12 * na by omega,
      Nat.mod_eq_of_lt hnaW]
  have ht10fr : ∀ i, i ≠ 0 → i ≠ 2 → t10.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi0]; exact ht9fr i hi0 hi2
  have ht10r15 : t10.reg 15 = some e := by
    rw [ht10fr 15 (by decide) (by decide)]; exact ht3r15
  have ht10r6 : t10.reg 6 = some c := by
    rw [ht10fr 6 (by decide) (by decide)]; exact ht3r6
  have ht10mem : t10.mem = m48 := by
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).mem = m48
    rw [mem_set]; exact ht9mem
  have ht10len : t10.regs.length = 16 := by
    rw [ht10]
    show (t9.set 0 (some (aluF .add (4 * na) (8 * na)).1)).regs.length =
      16
    rw [len_set]; exact ht9len
  have ht10k : t10.k = s.k := by rw [ht10]; exact ht9k
  have r10 : run stage1Im 1 (Out.ok t9) = Out.ok t10 := by
    rw [run_succ_ok stage1Im 0 t9, e10, ← ht10, run_zero]
  -- 0x3f9: mov rdx, r15
  have e11 : X86Int.step stage1Im t10 =
      .ok { t10.set 2 (some e) with rip := t10.rip + 3 } :=
    step_mov (by rw [ht10rip]; exact hc.p3f9) ht10r15
  obtain ⟨t11, ht11⟩ : ∃ t : X86Int.St, t =
      { t10.set 2 (some e) with rip := t10.rip + 3 } := ⟨_, rfl⟩
  have ht11rip : t11.rip = base + 0x3fc := by
    rw [ht11]; show t10.rip + 3 = base + 0x3fc; rw [ht10rip]
  have ht11r2 : t11.reg 2 = some e := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).reg 2 = some e
    rw [reg_with_rip, reg_set_self (by rw [ht10len]; decide)]
  have ht11fr : ∀ i, i ≠ 0 → i ≠ 2 → t11.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).reg i =
      t3.reg i
    rw [reg_with_rip, reg_set_other hi2]
    exact ht10fr i hi0 hi2
  have ht11r6 : t11.reg 6 = some c := by
    rw [ht11fr 6 (by decide) (by decide)]; exact ht3r6
  have ht11r0 : t11.reg 0 = some (12 * na) := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).reg 0 =
      some (12 * na)
    rw [reg_with_rip, reg_set_other (by decide)]; exact ht10r0
  have ht11mem : t11.mem = m48 := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).mem = m48
    rw [mem_with_rip, mem_set]; exact ht10mem
  have ht11len : t11.regs.length = 16 := by
    rw [ht11]
    show ({ t10.set 2 (some e) with rip := t10.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht10len
  have ht11k : t11.k = s.k := by rw [ht11]; exact ht10k
  have r11 : run stage1Im 1 (Out.ok t10) = Out.ok t11 := by
    rw [run_succ_ok stage1Im 0 t10, e11, ← ht11, run_zero]
  -- 0x3fc: sub rdx, rsi
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t11.rip + 3 } :=
    step_alu_sub (by rw [ht11rip]; exact hc.p3fc) ht11r2 ht11r6
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (aluF .sub e c).1) with
        fl := (aluF .sub e c).2, rip := t11.rip + 3 } := ⟨_, rfl⟩
  have hsub : (aluF .sub e c).1 = e - c := by
    rw [aluF_sub_fst, show e + W - c = (e - c) + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt (by omega)]
  have ht12rip : t12.rip = base + 0x3ff := by
    rw [ht12]; show t11.rip + 3 = base + 0x3ff; rw [ht11rip]
  have ht12r2 : t12.reg 2 = some (e - c) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).reg 2 = some (e - c)
    rw [reg_set_self (by rw [ht11len]; decide), hsub]
  have ht12r0 : t12.reg 0 = some (12 * na) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).reg 0 = some (12 * na)
    rw [reg_set_other (by decide)]; exact ht11r0
  have ht12fr : ∀ i, i ≠ 0 → i ≠ 2 → t12.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).reg i = t3.reg i
    rw [reg_set_other hi2]; exact ht11fr i hi0 hi2
  have ht12mem : t12.mem = m48 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).mem = m48
    rw [mem_set]; exact ht11mem
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .sub e c).1)).regs.length = 16
    rw [len_set]; exact ht11len
  have ht12k : t12.k = s.k := by rw [ht12]; exact ht11k
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  -- 0x3ff: cmp rdx, rax
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12 with fl := (aluF .sub (e - c) (12 * na)).2, rip := t12.rip + 3 } :=
    step_cmp (by rw [ht12rip]; exact hc.p3ff) ht12r2 ht12r0
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12 with fl := (aluF .sub (e - c) (12 * na)).2, rip := t12.rip + 3 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x402 := by
    rw [ht13]; show t12.rip + 3 = base + 0x402; rw [ht12rip]
  have ht13fl : X86Int.cond t13.fl 3 = some true := by
    have hcf : t13.fl.cf = some false := by
      rw [ht13]
      show (aluF .sub (e - c) (12 * na)).2.cf = some false
      exact aluF_sub_cf (by omega)
    have hcc := cond_cc3 hcf
    simpa using hcc
  have ht13fr : ∀ i, t13.reg i = t12.reg i := by
    intro i
    rw [ht13]
    show t12.reg i = t12.reg i
    exact rfl
  have ht13mem : t13.mem = m48 := by rw [ht13]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by rw [ht13]; exact ht12len
  have ht13k : t13.k = s.k := by rw [ht13]; exact ht12k
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  -- 0x402: jae 0x409
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with rip := base + 0x409 } := by
    have h := step_jcc (im := stage1Im) (s := t13)
      (by rw [ht13rip]; exact hc.p402) ht13fl
    simpa using h
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with rip := base + 0x409 } := ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x409 := by rw [ht14]
  have ht14fr : ∀ i, i ≠ 0 → i ≠ 2 → t14.reg i = t3.reg i := by
    intro i hi0 hi2
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).reg i = t3.reg i
    rw [reg_with_rip, ht13fr i]; exact ht12fr i hi0 hi2
  have ht14r6 : t14.reg 6 = some c := by
    rw [ht14fr 6 (by decide) (by decide)]; exact ht3r6
  have ht14r0 : t14.reg 0 = some (12 * na) := by
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).reg 0 = some (12 * na)
    rw [reg_with_rip, ht13fr 0]; exact ht12r0
  have ht14mem : t14.mem = m48 := by
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).mem = m48
    rw [mem_with_rip]; exact ht13mem
  have ht14len : t14.regs.length = 16 := by
    rw [ht14]
    show ({ t13 with rip := base + 0x409 }).regs.length = 16
    rw [len_with_rip]; exact ht13len
  have ht14k : t14.k = s.k := by rw [ht14]; exact ht13k
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  -- 0x409: add rsi, rax
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14.set 6 (some (aluF .add c (12 * na)).1) with
        fl := (aluF .add c (12 * na)).2, rip := t14.rip + 3 } :=
    step_alu_add (by rw [ht14rip]; exact hc.p409) ht14r6 ht14r0
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14.set 6 (some (aluF .add c (12 * na)).1) with
        fl := (aluF .add c (12 * na)).2, rip := t14.rip + 3 } :=
    ⟨_, rfl⟩
  have ht15rip : t15.rip = base + 0x40c := by
    rw [ht15]; show t14.rip + 3 = base + 0x40c; rw [ht14rip]
  have ht15r6 : t15.reg 6 = some (c + 12 * na) := by
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).reg 6 =
      some (c + 12 * na)
    rw [reg_set_self (by rw [ht14len]; decide), aluF_add_fst,
      Nat.mod_eq_of_lt (by omega)]
  have ht15fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → t15.reg i = t3.reg i := by
    intro i hi0 hi2 hi6
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).reg i = t3.reg i
    rw [reg_set_other hi6]; exact ht14fr i hi0 hi2
  have ht15r7 : t15.reg 7 = some p := by
    rw [ht15fr 7 (by decide) (by decide) (by decide)]; exact ht3r7
  have ht15r1 : t15.reg 1 = some (r + 1) := by
    rw [ht15fr 1 (by decide) (by decide) (by decide)]; exact ht3r1
  have ht15mem : t15.mem = m48 := by
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).mem = m48
    rw [mem_set]; exact ht14mem
  have ht15len : t15.regs.length = 16 := by
    rw [ht15]
    show (t14.set 6 (some (aluF .add c (12 * na)).1)).regs.length = 16
    rw [len_set]; exact ht14len
  have ht15k : t15.k = s.k := by rw [ht15]; exact ht14k
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  -- 0x40c: add rdi, 64
  have e16 : X86Int.step stage1Im t15 =
      .ok { t15.set 7 (some (aluF .add p (64 % W)).1) with
        fl := (aluF .add p (64 % W)).2, rip := t15.rip + 4 } :=
    step_aluI_add (by rw [ht15rip]; exact hc.p40c) ht15r7
  obtain ⟨t16, ht16⟩ : ∃ t : X86Int.St, t =
      { t15.set 7 (some (aluF .add p (64 % W)).1) with
        fl := (aluF .add p (64 % W)).2, rip := t15.rip + 4 } :=
    ⟨_, rfl⟩
  have ht16rip : t16.rip = base + 0x410 := by
    rw [ht16]; show t15.rip + 4 = base + 0x410; rw [ht15rip]
  have ht16r7 : t16.reg 7 = some (p + 64) := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg 7 =
      some (p + 64)
    rw [reg_set_self (by rw [ht15len]; decide), h64W, aluF_add_fst,
      Nat.mod_eq_of_lt hpW]
  have ht16r6 : t16.reg 6 = some (c + 12 * na) := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg 6 =
      some (c + 12 * na)
    rw [reg_set_other (by decide)]; exact ht15r6
  have ht16r1 : t16.reg 1 = some (r + 1) := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg 1 =
      some (r + 1)
    rw [reg_set_other (by decide)]; exact ht15r1
  have ht16fr : ∀ i, i ≠ 0 → i ≠ 2 → i ≠ 6 → i ≠ 7 →
      t16.reg i = t3.reg i := by
    intro i hi0 hi2 hi6 hi7
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).reg i = t3.reg i
    rw [reg_set_other hi7]; exact ht15fr i hi0 hi2 hi6
  have ht16mem : t16.mem = m48 := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).mem = m48
    rw [mem_set]; exact ht15mem
  have ht16len : t16.regs.length = 16 := by
    rw [ht16]
    show (t15.set 7 (some (aluF .add p (64 % W)).1)).regs.length = 16
    rw [len_set]; exact ht15len
  have ht16k : t16.k = s.k := by rw [ht16]; exact ht15k
  have r16 : run stage1Im 1 (Out.ok t15) = Out.ok t16 := by
    rw [run_succ_ok stage1Im 0 t15, e16, ← ht16, run_zero]
  -- 0x410: sub rcx, 1
  have e17 : X86Int.step stage1Im t16 =
      .ok { t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1) with
        fl := (aluF .sub (r + 1) (1 % W)).2, rip := t16.rip + 4 } :=
    step_aluI_sub (by rw [ht16rip]; exact hc.p410) ht16r1
  obtain ⟨t17, ht17⟩ : ∃ t : X86Int.St, t =
      { t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1) with
        fl := (aluF .sub (r + 1) (1 % W)).2, rip := t16.rip + 4 } :=
    ⟨_, rfl⟩
  have hsubr : (aluF .sub (r + 1) (1 % W)).1 = r := by
    rw [h1W, aluF_sub_fst, show r + 1 + W - 1 = r + W by omega,
      Nat.add_mod_right, Nat.mod_eq_of_lt hrW]
  have ht17rip : t17.rip = base + 0x414 := by
    rw [ht17]; show t16.rip + 4 = base + 0x414; rw [ht16rip]
  have ht17r1 : t17.reg 1 = some r := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg 1 = some r
    rw [reg_set_self (by rw [ht16len]; decide), hsubr]
  have ht17r6 : t17.reg 6 = some (c + 12 * na) := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg 6 =
      some (c + 12 * na)
    rw [reg_set_other (by decide)]; exact ht16r6
  have ht17r7 : t17.reg 7 = some (p + 64) := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg 7 =
      some (p + 64)
    rw [reg_set_other (by decide)]; exact ht16r7
  have ht17fr : ∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 6 → i ≠ 7 →
      t17.reg i = t3.reg i := by
    intro i hi0 hi1 hi2 hi6 hi7
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).reg i =
      t3.reg i
    rw [reg_set_other hi1]; exact ht16fr i hi0 hi2 hi6 hi7
  have ht17mem : t17.mem = m48 := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).mem = m48
    rw [mem_set]; exact ht16mem
  have ht17len : t17.regs.length = 16 := by
    rw [ht17]
    show (t16.set 1 (some (aluF .sub (r + 1) (1 % W)).1)).regs.length =
      16
    rw [len_set]; exact ht16len
  have ht17k : t17.k = s.k := by rw [ht17]; exact ht16k
  have r17 : run stage1Im 1 (Out.ok t16) = Out.ok t17 := by
    rw [run_succ_ok stage1Im 0 t16, e17, ← ht17, run_zero]
  -- 0x414: jmp word-loop head
  obtain ⟨t18, hrun18, ht18rip, ht18fr, ht18mem, ht18len, ht18k⟩ :=
    jmp_step hc.p414 ht17rip
  have hrun : run stage1Im 18 (Out.ok s) = Out.ok t18 := by
    rw [show (18 : Nat) = 1 + 17 by omega, run_add stage1Im 1 17, r1,
      show (17 : Nat) = 1 + 16 by omega, run_add stage1Im 1 16, hrun2,
      show (16 : Nat) = 1 + 15 by omega, run_add stage1Im 1 15, r3,
      show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r4,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r5,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r6,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r7,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r8,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r9,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r10,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r11,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r12,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r13,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r14,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r15,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r16,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r17,
      hrun18]
  have ht18mem' : t18.mem = m48 := by rw [ht18mem]; exact ht17mem
  refine ⟨t18, hrun, ht18rip, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht18fr 6]; exact ht17r6
  · rw [ht18fr 7]; exact ht17r7
  · rw [ht18fr 1]; exact ht17r1
  · rw [ht18fr 4, ht17fr 4 (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact ht3r4
  · rw [ht18mem']
    have h : rd64 m48 (p + 48) = some (c % W) := by
      rw [ht2mem] at ht2slot
      exact ht2slot
    exact h
  · rw [ht18mem']
    have h := rd64_wr64_other hm48 (m := s.mem) (x := p + 32)
      (Or.inr (by omega))
    rw [h]; exact hslot32
  · intro i hi0 hi1 hi2 hi4 hi6 hi7
    rw [ht18fr i, ht17fr i hi0 hi1 hi2 hi6 hi7]
    exact ht3fr i hi0 hi1 hi2 hi4 hi6 hi7
  · rw [ht18len]; exact ht17len
  · rw [ht18k]; exact ht17k
  · rw [ht18mem']; exact wrN_wshape _ _ _ hm48
  · intro x k hd
    rw [ht18mem']
    exact rdN_wr64_other hm48 hd

/-- `rd` equality from an `rdN _ 1` equality. -/
theorem rd_eq_of_rdN1 {m m' : Mem} {a : Nat}
    (h : m'.rdN a 1 = m.rdN a 1) : m'.rd a = m.rd a := by
  cases h1 : m.rd a with
  | none =>
    cases h2 : m'.rd a with
    | none => rfl
    | some v =>
      have e1 : m.rdN a 1 = none := by simp [Mem.rdN, h1]
      have e2 : m'.rdN a 1 = some [v] := by simp [Mem.rdN, h2]
      rw [e2, e1] at h
      exact absurd h (by simp)
  | some v =>
    exact rd_of_rdN1 h h1

/-- `LenLay` transport along point-wise `rd` equality on the
scratch range. -/
theorem LenLay_of_frame : ∀ (code : List Instr) {m m' : Mem}
    {B S c : Nat}, B ≤ c → LenLay m B S c code →
    (∀ a, S + (c - B) ≤ a → a < S + (c + codeBytes code - B) →
      m'.rd a = m.rd a) →
    LenLay m' B S c code := by
  intro code
  induction code with
  | nil => intro m m' B S c _ _ _; trivial
  | cons x xs ih =>
    intro m m' B S c hBc hlay hfr
    obtain ⟨h0, hrest⟩ := hlay
    have hpos : 0 < x.enc.length := by
      obtain ⟨op, rest, heq, _⟩ := enc_head x
      rw [heq]; exact Nat.zero_lt_succ _
    constructor
    · rw [hfr (S + (c - B)) (by omega) (by
        show S + (c - B) <
          S + (c + codeBytes (x :: xs) - B)
        have hcb : codeBytes (x :: xs) =
          x.enc.length + codeBytes xs := rfl
        omega)]
      exact h0
    · apply ih (by omega) hrest
      intro a ha1 ha2
      apply hfr
      · omega
      · have hcb : codeBytes (x :: xs) =
          x.enc.length + codeBytes xs := rfl
        omega

/-- One full word of the parser loop: fields, code loop, ann
skip, entry advance, back to the word-loop head with one fewer
word remaining. -/
theorem parser_word_step {B e S sp p c r : Nat} {wd : Word}
    {bs : List Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x2d6) (hr1 : s.reg 1 = some (r + 1))
    (hr6 : s.reg 6 = some c) (hr15 : s.reg 15 = some e)
    (hr7 : s.reg 7 = some p) (hr4 : s.reg 4 = some sp)
    (hr3 : s.reg 3 = some B) (hr11 : s.reg 11 = some S)
    (h20 : s.mem.rdN c 20 = some bs)
    (heq0 : leVal (bs.take 4) % W = wd.ins)
    (heq1 : leVal ((bs.drop 4).take 4) % W = wd.outs)
    (heq2 : leVal ((bs.drop 8).take 4) % W = wd.locals)
    (heq3 : leVal ((bs.drop 12).take 4) % W = wd.code.length)
    (heq4 : leVal ((bs.drop 16).take 4) % W = wd.ann.length)
    (hseq : CodeSeq s.mem (c + 20) wd.code)
    (htot : c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤ e)
    (hcW : c < W) (heW : e < W) (hpW : p + 64 < W)
    (hr1W : r + 1 < W) (hBc : B ≤ c)
    (hSW : S + (c + 20 + codeBytes wd.code - B) < W)
    (hroom : StackRoom s 2) (hsp16 : 16 ≤ sp) (hspW : sp < W)
    (hlen16 : s.regs.length = 16)
    (hEB : p + 64 ≤ c ∨ e ≤ p)
    (hSB : e ≤ sp - 16 ∨ sp ≤ c)
    (hES : p + 64 ≤ sp - 16 ∨ sp ≤ p)
    (hScrB : S + (c + 20 + codeBytes wd.code - B) ≤ c + 20 ∨
      c + 20 + codeBytes wd.code + 12 * wd.ann.length ≤ S + (c + 20 - B))
    (hScrE : S + (c + 20 + codeBytes wd.code - B) ≤ p ∨
      p + 64 ≤ S + (c + 20 - B))
    (hScrStk : S + (c + 20 + codeBytes wd.code - B) ≤ sp - 8 ∨
      sp ≤ S + (c + 20 - B))
    (hwrE : ∀ (off v : Nat), off < 64 →
      (wr64 s.mem (p + off) v).isSome)
    (hwrS : ∀ a v, S + (c + 20 - B) ≤ a →
      a < S + (c + 20 + codeBytes wd.code - B) →
      (s.mem.wr a v).isSome) :
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some r ∧
      s'.reg 6 = some (c + 20 + codeBytes wd.code +
        12 * wd.ann.length) ∧
      s'.reg 7 = some (p + 64) ∧ s'.reg 4 = some sp ∧
      rd64 s'.mem p = some wd.ins ∧
      rd64 s'.mem (p + 8) = some wd.outs ∧
      rd64 s'.mem (p + 16) = some wd.locals ∧
      rd64 s'.mem (p + 24) = some wd.code.length ∧
      rd64 s'.mem (p + 32) = some wd.ann.length ∧
      rd64 s'.mem (p + 40) = some ((c + 20) % W) ∧
      rd64 s'.mem (p + 48) =
        some ((c + 20 + codeBytes wd.code) % W) ∧
      CodeSeq s'.mem (c + 20) wd.code ∧
      LenLay s'.mem B S (c + 20) wd.code ∧
      s'.mem.rdN (c + 20 + codeBytes wd.code) (12 * wd.ann.length) =
        s.mem.rdN (c + 20 + codeBytes wd.code) (12 * wd.ann.length) ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 ≤ x ∨ x + k ≤ p) →
        (S + (c + 20 + codeBytes wd.code - B) ≤ x ∨
          x + k ≤ S + (c + 20 - B)) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  -- fields
  obtain ⟨nf, s1, hrunf, hs1rip, hs1r6, hs1r4, hs1r7, hs1s0, hs1s8,
      hs1s16, hs1s24, hs1s32, hs1s40, hs1fr, hs1len, hs1k, hs1sh,
      hs1room, hs1h20, hs1wr, hs1memfr⟩ :=
    parser_word_fields hrip hr6 hr15 hr7 hr4 h20 (by omega) hcW heW
      hpW hroom hsp16 hspW hlen16 hEB hSB hES hwrE
  have hs1seq : CodeSeq s1.mem (c + 20) wd.code := by
    apply CodeSeq_of_frame hseq
    intro y k hy hk
    apply hs1memfr <;> omega
  have hs1ann : s1.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
      s.mem.rdN (c + 20 + codeBytes wd.code) (12 * wd.ann.length) :=
    hs1memfr _ _ (by omega) (by omega)
  have hs1r1 : s1.reg 1 = some (r + 1) := by
    rw [hs1fr 1 (by decide) (by decide) (by decide) (by decide)]
    exact hr1
  have hs1s24' : rd64 s1.mem (p + 24) = some wd.code.length := by
    rw [hs1s24, heq3]
  have hroom1 : StackRoom s1 1 := by
    intro spx hspx i hi v
    exact hs1room spx hspx i (by omega) v
  -- enter
  obtain ⟨s2, hrun2, hs2rip, hs2r1, hs2r4, hs2pop, hs2s24, hs2fr,
      hs2len, hs2k, hs2sh, hs2memfr⟩ :=
    parser_code_enter hs1rip hs1r1 hs1r7 hs1r4 hroom1 (by omega)
      hspW (by omega) hs1s24' (by omega) hs1len
  have hs2seq : CodeSeq s2.mem (c + 20) wd.code := by
    apply CodeSeq_of_frame hs1seq
    intro y k hy hk
    apply hs2memfr
    omega
  have hs2ann : s2.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
      s.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) := by
    rw [hs2memfr _ _ (by omega)]; exact hs1ann
  have hs2r6 : s2.reg 6 = some (c + 20) := by
    rw [hs2fr 6 (by decide) (by decide)]; exact hs1r6
  have hs2r15 : s2.reg 15 = some e := by
    rw [hs2fr 15 (by decide) (by decide),
      hs1fr 15 (by decide) (by decide) (by decide) (by decide)]
    exact hr15
  have hs2r3 : s2.reg 3 = some B := by
    rw [hs2fr 3 (by decide) (by decide),
      hs1fr 3 (by decide) (by decide) (by decide) (by decide)]
    exact hr3
  have hs2r11 : s2.reg 11 = some S := by
    rw [hs2fr 11 (by decide) (by decide),
      hs1fr 11 (by decide) (by decide) (by decide) (by decide)]
    exact hr11
  have hs2sh' : wshape s2.mem = wshape s.mem := by rw [hs2sh, hs1sh]
  have hs2wr : ∀ a v, S + (c + 20 - B) ≤ a →
      a < S + (c + 20 + codeBytes wd.code - B) →
      (s2.mem.wr a v).isSome := by
    intro a v ha1 ha2
    have h1 : (s.mem.wrN a [v]).isSome := by
      rw [← wr_eq_wrN]; exact hwrS a v ha1 ha2
    have h2 := wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
    rwa [← wr_eq_wrN] at h2
  -- code loop
  obtain ⟨k3, s3, hrun3, hs3rip, hs3r1, hs3r6, hs3seq, hs3lay, hs3fr,
      hs3len, hs3k, hs3sh, hs3memfr⟩ :=
    parser_code_loop wd.code hs2rip hs2r1 hs2r6 hs2r15 hs2r3 hs2r11
      hs2seq (by omega) (by omega) heW (by omega) hSW (by omega)
      hs2wr hs2len
  have hs3ann : s3.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
      s.mem.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) := by
    rw [hs3memfr _ _ (by omega)]; exact hs2ann
  have slot3 : ∀ {off w : Nat}, off + 8 ≤ 48 →
      rd64 s2.mem (p + off) = some w → rd64 s3.mem (p + off) = some w := by
    intro off w hoff h
    unfold rd64
    rw [hs3memfr (p + off) 8 (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  have slot2 : ∀ {off w : Nat}, off + 8 ≤ 64 →
      rd64 s1.mem (p + off) = some w → rd64 s2.mem (p + off) = some w := by
    intro off w _ h
    unfold rd64
    rw [hs2memfr (p + off) 8 (by omega)]
    have h' := h
    unfold rd64 at h'
    exact h'
  have hs3s32 : rd64 s3.mem (p + 32) = some wd.ann.length :=
    slot3 (by decide) (slot2 (by decide) (by rw [hs1s32, heq4]))
  have hs3pop : rd64 s3.mem (sp - 8) = some (r + 1) := by
    unfold rd64
    rw [hs3memfr (sp - 8) 8 (by omega)]
    have h := hs2pop
    unfold rd64 at h
    rwa [Nat.mod_eq_of_lt hr1W] at h
  -- loop exit
  obtain ⟨s4, hrun4, hs4rip, hs4fr, hs4mem, hs4len, hs4k⟩ :=
    parser_code_head_zero hs3rip hs3r1
  have hs4r4 : s4.reg 4 = some (sp - 8) := by
    rw [hs4fr 4, hs3fr 4 (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hs2r4
  have hs4r6 : s4.reg 6 = some (c + 20 + codeBytes wd.code) := by
    rw [hs4fr 6]; exact hs3r6
  have hs4r7 : s4.reg 7 = some p := by
    rw [hs4fr 7, hs3fr 7 (by decide) (by decide) (by decide)
      (by decide) (by decide), hs2fr 7 (by decide) (by decide)]
    exact hs1r7
  have hs4r15 : s4.reg 15 = some e := by
    rw [hs4fr 15, hs3fr 15 (by decide) (by decide) (by decide)
      (by decide) (by decide)]
    exact hs2r15
  have hs4pop : rd64 s4.mem (sp - 8) = some (r + 1) := by
    rw [hs4mem]; exact hs3pop
  have hs4s32 : rd64 s4.mem (p + 32) = some wd.ann.length := by
    rw [hs4mem]; exact hs3s32
  have hs4sh : wshape s4.mem = wshape s1.mem := by
    rw [hs4mem, hs3sh, hs2sh]
  obtain ⟨m48, hm48⟩ : ∃ m, wr64 s4.mem (p + 48)
      (c + 20 + codeBytes wd.code) = some m := by
    have h1 : (s1.mem.wrN (p + 48)
        (bytes8 (c + 20 + codeBytes wd.code))).isSome :=
      hs1wr 48 _ (by decide)
    have h2 := wrN_some_of_wshape _ s1.mem s4.mem hs4sh _ h1
    exact Option.isSome_iff_exists.mp h2
  -- word tail
  obtain ⟨s5, hrun5, hs5rip, hs5r6, hs5r7, hs5r1, hs5r4, hs5s48,
      hs5s32, hs5fr, hs5len, hs5k, hs5sh, hs5memfr⟩ :=
    parser_word_tail hs4rip hs4r4 hs4r6 hs4r7 hs4r15 hs4pop
      (by omega) (by omega) (by omega) heW hpW (by omega) (by omega)
      hs4s32 hm48 (by rw [hs4len]; exact hs3len)
  have hrun : run stage1Im (nf + 2 + k3 + 3 + 18) (Out.ok s) =
      Out.ok s5 := by
    rw [show nf + 2 + k3 + 3 + 18 = nf + (2 + (k3 + (3 + 18))) by omega,
      run_add stage1Im nf _, hrunf, run_add stage1Im 2 _, hrun2,
      run_add stage1Im k3 _, hrun3, run_add stage1Im 3 18, hrun4, hrun5]
  have slot5 : ∀ {off w : Nat}, off + 8 ≤ 48 →
      rd64 s1.mem (p + off) = some w → rd64 s5.mem (p + off) = some w := by
    intro off w hoff h
    have h3 : rd64 s3.mem (p + off) = some w :=
      slot3 hoff (slot2 (by omega) h)
    have h4 : rd64 s4.mem (p + off) = some w := by rw [hs4mem]; exact h3
    unfold rd64
    rw [hs5memfr (p + off) 8 (by omega)]
    have h' := h4
    unfold rd64 at h'
    exact h'
  have hs5sh' : wshape s5.mem = wshape s.mem := by
    rw [hs5sh, hs4mem, hs3sh, hs2sh, hs1sh]
  have hroom5 : StackRoom s5 2 := by
    apply StackRoom_of_wshape hs5sh' _ hroom
    have h1 : sp - 8 + 8 = sp := by omega
    rw [show s5.reg 4 = some sp by rw [hs5r4, h1], hr4]
  refine ⟨nf + 2 + k3 + 3 + 18, s5, hrun, hs5rip, hs5r1, hs5r6, hs5r7,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, hs5s48, ?_, ?_, ?_, ?_, hs5len, ?_,
    hs5sh', hroom5, ?_⟩
  · have h1 : sp - 8 + 8 = sp := by omega
    rw [hs5r4, h1]
  · have h := slot5 (off := 0) (by decide) (by simpa using hs1s0)
    rwa [heq0] at h
  · have h := slot5 (by decide) hs1s8
    rwa [heq1] at h
  · have h := slot5 (by decide) hs1s16
    rwa [heq2] at h
  · have h := slot5 (by decide) hs1s24
    rwa [heq3] at h
  · exact hs5s32
  · exact slot5 (by decide) hs1s40
  · have h1 : s5.mem.rdN (c + 20) (codeBytes wd.code) =
        s3.mem.rdN (c + 20) (codeBytes wd.code) := by
      rw [hs5memfr _ _ (by omega), hs4mem]
    exact CodeSeq_of_frame hs3seq (fun y k hy hk => by
      have e1 : s5.mem.rdN y k = s4.mem.rdN y k :=
        hs5memfr y k (by omega)
      rw [e1, hs4mem])
  · apply LenLay_of_frame wd.code (by omega) hs3lay
    intro a ha1 ha2
    have e1 : s5.mem.rd a = s4.mem.rd a :=
      rd_eq_of_rdN1 (hs5memfr a 1 (by omega))
    rw [e1, hs4mem]
  · rw [hs5memfr _ _ (by omega), hs4mem]; exact hs3ann
  · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
    rw [hs5fr i hi0 hi1 hi2 hi4 hi6 hi7, hs4fr i,
      hs3fr i hi0 hi1 hi2 hi6 hi10,
      hs2fr i hi1 hi4]
    exact hs1fr i hi0 hi2 hi4 hi6
  · rw [hs5k, hs4k, hs3k, hs2k, hs1k]
  · intro x k hd1 hd2 hd3
    rw [hs5memfr x k (by omega), hs4mem, hs3memfr x k hd3,
      hs2memfr x k (by omega), hs1memfr x k hd1 hd2]

/-- Buffer bytes consumed by a word list. -/
def wordsBytes : List Word → Nat
  | [] => 0
  | wd :: wds => (20 + codeBytes wd.code + 12 * wd.ann.length) +
      wordsBytes wds

/-- Memory-level annotation layout: entry `i` at `p + 12*i`
holds `(index, d, a)` as three `u32`s. -/
def AnnLayB (m : Mem) (p : Nat) (ann : List (Nat × Height)) : Prop :=
  ∀ i (hi : i < ann.length),
    (m.rdN (p + 12 * i) 4).map leVal = some ann[i].1 ∧
    (m.rdN (p + 12 * i + 4) 4).map leVal = some ann[i].2.d ∧
    (m.rdN (p + 12 * i + 8) 4).map leVal = some ann[i].2.a ∧
    ann[i].1 < W

theorem AnnLayB_of_frame {m m' : Mem} {p : Nat}
    {ann : List (Nat × Height)}
    (h : AnnLayB m p ann)
    (hfr : ∀ x k, p ≤ x → x + k ≤ p + 12 * ann.length →
      m'.rdN x k = m.rdN x k) : AnnLayB m' p ann := by
  intro i hi
  obtain ⟨h1, h2, h3, h4⟩ := h i hi
  refine ⟨?_, ?_, ?_, h4⟩
  · rw [hfr (p + 12 * i) 4 (by omega) (by omega)]; exact h1
  · rw [hfr (p + 12 * i + 4) 4 (by omega) (by omega)]; exact h2
  · rw [hfr (p + 12 * i + 8) 4 (by omega) (by omega)]; exact h3

/-- What the parser needs laid out in the buffer for a word
list: header fields, code encodings and annotation arrays. -/
def WordsLay (m : Mem) (c : Nat) : List Word → Prop
  | [] => True
  | wd :: wds =>
    (∃ bs, m.rdN c 20 = some bs ∧
      leVal (bs.take 4) % W = wd.ins ∧
      leVal ((bs.drop 4).take 4) % W = wd.outs ∧
      leVal ((bs.drop 8).take 4) % W = wd.locals ∧
      leVal ((bs.drop 12).take 4) % W = wd.code.length ∧
      leVal ((bs.drop 16).take 4) % W = wd.ann.length) ∧
    CodeSeq m (c + 20) wd.code ∧
    AnnLayB m (c + 20 + codeBytes wd.code) wd.ann ∧
    WordsLay m (c + 20 + codeBytes wd.code + 12 * wd.ann.length) wds

theorem WordsLay_of_frame : ∀ (wds : List Word) {m m' : Mem}
    {c : Nat}, WordsLay m c wds →
    (∀ x k, c ≤ x → x + k ≤ c + wordsBytes wds →
      m'.rdN x k = m.rdN x k) →
    WordsLay m' c wds := by
  intro wds
  induction wds with
  | nil => intro m m' c _ _; trivial
  | cons wd wds ih =>
    intro m m' c h hfr
    obtain ⟨⟨bs, h20, e0, e1, e2, e3, e4⟩, hseq, hann, hrest⟩ := h
    have hcb : wordsBytes (wd :: wds) =
        (20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds := rfl
    refine ⟨⟨bs, ?_, e0, e1, e2, e3, e4⟩, ?_, ?_, ?_⟩
    · rw [hfr c 20 (by omega) (by omega)]; exact h20
    · apply CodeSeq_of_frame hseq
      intro y k hy hk
      apply hfr <;> omega
    · apply AnnLayB_of_frame hann
      intro y k hy hk
      apply hfr <;> omega
    · apply ih hrest
      intro y k hy hk
      apply hfr <;> omega

/-- What the parser leaves in the words table, code buffer,
length scratch and ann arrays for a word list. -/
def TabLay (m : Mem) (B S tbl c : Nat) : List Word → Prop
  | [] => True
  | wd :: wds =>
    rd64 m tbl = some wd.ins ∧
    rd64 m (tbl + 8) = some wd.outs ∧
    rd64 m (tbl + 16) = some wd.locals ∧
    rd64 m (tbl + 24) = some wd.code.length ∧
    rd64 m (tbl + 32) = some wd.ann.length ∧
    rd64 m (tbl + 40) = some ((c + 20) % W) ∧
    rd64 m (tbl + 48) = some ((c + 20 + codeBytes wd.code) % W) ∧
    CodeSeq m (c + 20) wd.code ∧
    LenLay m B S (c + 20) wd.code ∧
    AnnLayB m (c + 20 + codeBytes wd.code) wd.ann ∧
    TabLay m B S (tbl + 64)
      (c + 20 + codeBytes wd.code + 12 * wd.ann.length) wds

set_option maxHeartbeats 0 in
/-- The whole parser word loop over a word list: from the
word-loop head back to the word-loop head with count zero and
the full table laid out. -/
theorem parser_words_loop : ∀ (wds : List Word)
    {B e S sp p c : Nat} {s : X86Int.St},
    s.rip = base + 0x2cc → s.reg 1 = some wds.length →
    s.reg 6 = some c → s.reg 7 = some p →
    s.reg 15 = some e → s.reg 4 = some sp →
    s.reg 3 = some B → s.reg 11 = some S →
    WordsLay s.mem c wds →
    B ≤ c → c + wordsBytes wds ≤ e → e < W →
    p + 64 * wds.length < W → S + (c + wordsBytes wds - B) < W →
    wds.length < W →
    StackRoom s 2 → 16 ≤ sp → sp < W → s.regs.length = 16 →
    (p + 64 * wds.length ≤ B ∨ e ≤ p) →
    (p + 64 * wds.length ≤ sp - 16 ∨ sp ≤ p) →
    (p + 64 * wds.length ≤ S + (c - B) ∨
      S + (c + wordsBytes wds - B) ≤ p) →
    (e ≤ sp - 16 ∨ sp ≤ B) →
    (S + (c + wordsBytes wds - B) ≤ c ∨
      c + wordsBytes wds ≤ S + (c - B)) →
    (S + (c + wordsBytes wds - B) ≤ sp - 16 ∨ sp ≤ S + (c - B)) →
    (∀ (off v : Nat), off < 64 * wds.length →
      (wr64 s.mem (p + off) v).isSome) →
    (∀ a v, S + (c - B) ≤ a → a < S + (c + wordsBytes wds - B) →
      (s.mem.wr a v).isSome) →
    ∃ k s', run stage1Im k (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x2cc ∧ s'.reg 1 = some 0 ∧
      s'.reg 6 = some (c + wordsBytes wds) ∧
      s'.reg 7 = some (p + 64 * wds.length) ∧
      s'.reg 4 = some sp ∧
      TabLay s'.mem B S p c wds ∧
      (∀ i, i ≠ 0 → i ≠ 1 → i ≠ 2 → i ≠ 4 → i ≠ 6 → i ≠ 7 → i ≠ 10 →
        s'.reg i = s.reg i) ∧
      s'.regs.length = 16 ∧ s'.k = s.k ∧
      wshape s'.mem = wshape s.mem ∧ StackRoom s' 2 ∧
      (∀ x k, (x + k ≤ sp - 16 ∨ sp ≤ x) →
        (p + 64 * wds.length ≤ x ∨ x + k ≤ p) →
        (S + (c + wordsBytes wds - B) ≤ x ∨ x + k ≤ S + (c - B)) →
        s'.mem.rdN x k = s.mem.rdN x k) := by
  intro wds
  induction wds with
  | nil =>
    intro B e S sp p c s hrip hr1 hr6 hr7 _ hr4 _ _ _ _ _ _ _ _ _
      hroom _ _ hlen16 _ _ _ _ _ _ _ _
    refine ⟨0, s, rfl, hrip, hr1, ?_, ?_, hr4, trivial, ?_, hlen16,
      rfl, rfl, hroom, ?_⟩
    · show s.reg 6 = some (c + wordsBytes [])
      exact hr6
    · show s.reg 7 = some (p + 64 * [].length)
      exact hr7
    · intro i _ _ _ _ _ _ _; exact rfl
    · intro x k _ _ _; exact rfl
  | cons wd wds ih =>
    intro B e S sp p c s hrip hr1 hr6 hr7 hr15 hr4 hr3 hr11 hlay hBc
      htot heW hpW hSW hlenW hroom hsp16 hspW hlen16 hTB hTStk hTScr
      hBStk hBScr hScrStk hwrE hwrS
    obtain ⟨⟨bs, h20, e0, e1, e2, e3, e4⟩, hseq, hann, hrest⟩ := hlay
    have hwb : wordsBytes (wd :: wds) =
        (20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds := rfl
    have hlen' : wds.length + 1 = (wd :: wds).length := rfl
    have hcW : c < W := by omega
    -- head: enter the body
    obtain ⟨s1, hrun1, hs1rip, hs1fr, hs1mem, hs1len, hs1k⟩ :=
      parser_words_head_pos hrip hr1 (by omega) hlenW
    have hs1len16 : s1.regs.length = 16 := by rw [hs1len]; exact hlen16
    have hs1r : ∀ i v, s.reg i = some v → s1.reg i = some v := by
      intro i v h
      rw [hs1fr i]; exact h
    -- one word step
    obtain ⟨k1, s2, hrun2, hs2rip, hs2r1, hs2r6, hs2r7, hs2r4, hs2s0,
        hs2s8, hs2s16, hs2s24, hs2s32, hs2s40, hs2s48, hs2seq, hs2lay,
        hs2ann, hs2fr, hs2len, hs2k, hs2sh, hs2room, hs2memfr⟩ :=
      parser_word_step hs1rip (hs1r 1 _ hr1) (hs1r 6 _ hr6)
        (hs1r 15 _ hr15) (hs1r 7 _ hr7) (hs1r 4 _ hr4) (hs1r 3 _ hr3)
        (hs1r 11 _ hr11) (by rw [hs1mem]; exact h20) e0 e1 e2 e3 e4
        (by rw [hs1mem]; exact hseq) (by omega) hcW heW (by omega)
        (by omega) hBc (by omega)
        (StackRoom_of_mem hs1mem (hs1fr 4) hroom)
        hsp16 hspW hs1len16 (by omega) (by omega) (by omega)
        (by omega) (by omega) (by omega)
        (by
          intro off v hoff
          rw [hs1mem]
          exact hwrE off v (by omega))
        (by
          intro a v ha1 ha2
          rw [hs1mem]
          exact hwrS a v (by omega) (by omega))
    have hs2sh' : wshape s2.mem = wshape s.mem := by
      rw [hs2sh, hs1mem]
    have hs2lay' : WordsLay s2.mem
        (c + 20 + codeBytes wd.code + 12 * wd.ann.length) wds := by
      apply WordsLay_of_frame wds hrest
      intro y k hy hk
      rw [hs2memfr y k (by omega) (by omega) (by omega), hs1mem]
    have hs2wrE : ∀ (off v : Nat), off < 64 * wds.length →
        (wr64 s2.mem (p + 64 + off) v).isSome := by
      intro off v hoff
      have h1 : (s.mem.wrN (p + 64 + off) (bytes8 v)).isSome := by
        have h2 := hwrE (64 + off) v (by omega)
        rwa [show p + (64 + off) = p + 64 + off by omega] at h2
      exact wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
    have hs2wrS : ∀ a v,
        S + (c + 20 + codeBytes wd.code + 12 * wd.ann.length - B) ≤ a →
        a < S + (c + 20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds - B) →
        (s2.mem.wr a v).isSome := by
      intro a v ha1 ha2
      have h1 : (s.mem.wrN a [v]).isSome := by
        rw [← wr_eq_wrN]
        exact hwrS a v (by omega) (by omega)
      have h2 := wrN_some_of_wshape _ s.mem s2.mem hs2sh' _ h1
      rwa [← wr_eq_wrN] at h2
    have hs2r15 : s2.reg 15 = some e := by
      rw [hs2fr 15 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hs1fr 15]
      exact hr15
    have hs2r3 : s2.reg 3 = some B := by
      rw [hs2fr 3 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hs1fr 3]
      exact hr3
    have hs2r11 : s2.reg 11 = some S := by
      rw [hs2fr 11 (by decide) (by decide) (by decide) (by decide)
        (by decide) (by decide) (by decide), hs1fr 11]
      exact hr11
    -- induction over the rest
    obtain ⟨k2, s3, hrun3, hs3rip, hs3r1, hs3r6, hs3r7, hs3r4, hs3tab,
        hs3fr, hs3len, hs3k, hs3sh, hs3room, hs3memfr⟩ :=
      ih hs2rip hs2r1 hs2r6 hs2r7 hs2r15 hs2r4 hs2r3 hs2r11 hs2lay'
        (by omega) (by omega) heW (by omega) (by omega) (by omega)
        hs2room hsp16 hspW hs2len (by omega) (by omega) (by omega)
        hBStk (by omega) (by omega) hs2wrE hs2wrS
    have hrun : run stage1Im (2 + k1 + k2) (Out.ok s) = Out.ok s3 := by
      rw [show 2 + k1 + k2 = 2 + (k1 + k2) by omega,
        run_add stage1Im 2 _, hrun1, run_add stage1Im k1 k2, hrun2,
        hrun3]
    have slot3 : ∀ {off w : Nat}, off + 8 ≤ 56 →
        rd64 s2.mem (p + off) = some w →
        rd64 s3.mem (p + off) = some w := by
      intro off w hoff h
      unfold rd64
      rw [hs3memfr (p + off) 8 (by omega) (by omega) (by omega)]
      have h' := h
      unfold rd64 at h'
      exact h'
    have hseq3 : CodeSeq s3.mem (c + 20) wd.code := by
      apply CodeSeq_of_frame hs2seq
      intro y k hy hk
      apply hs3memfr <;> omega
    have hlay3 : LenLay s3.mem B S (c + 20) wd.code := by
      apply LenLay_of_frame wd.code (by omega) hs2lay
      intro a ha1 ha2
      exact rd_eq_of_rdN1 (hs3memfr a 1 (by omega) (by omega)
        (by omega))
    have hann2 : AnnLayB s2.mem (c + 20 + codeBytes wd.code)
        wd.ann := by
      apply AnnLayB_of_frame hann
      intro y k hy hk
      rw [hs2memfr y k (by omega) (by omega) (by omega), hs1mem]
    have hann3 : AnnLayB s3.mem (c + 20 + codeBytes wd.code)
        wd.ann := by
      apply AnnLayB_of_frame hann2
      intro y k hy hk
      apply hs3memfr <;> omega
    have htab : TabLay s3.mem B S p c (wd :: wds) :=
      ⟨slot3 (off := 0) (by decide) (by simpa using hs2s0),
        slot3 (by decide) hs2s8, slot3 (by decide) hs2s16,
        slot3 (by decide) hs2s24, slot3 (by decide) hs2s32,
        slot3 (by decide) hs2s40, slot3 (by decide) hs2s48,
        hseq3, hlay3, hann3, hs3tab⟩
    refine ⟨2 + k1 + k2, s3, hrun, hs3rip, hs3r1, ?_, ?_, hs3r4, htab,
      ?_, hs3len, ?_, ?_, hs3room, ?_⟩
    · show s3.reg 6 = some (c + wordsBytes (wd :: wds))
      rw [hwb]
      have h := hs3r6
      rwa [show c + 20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds =
        c + (20 + codeBytes wd.code + 12 * wd.ann.length +
          wordsBytes wds) by omega] at h
    · show s3.reg 7 = some (p + 64 * (wd :: wds).length)
      have h := hs3r7
      rwa [show p + 64 + 64 * wds.length =
        p + 64 * (wd :: wds).length by
        rw [← hlen']; omega] at h
    · intro i hi0 hi1 hi2 hi4 hi6 hi7 hi10
      rw [hs3fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10,
        hs2fr i hi0 hi1 hi2 hi4 hi6 hi7 hi10, hs1fr i]
    · rw [hs3k, hs2k, hs1k]
    · rw [hs3sh, hs2sh, hs1mem]
    · intro x k hd1 hd2 hd3
      rw [hs3memfr x k hd1 (by omega) (by omega),
        hs2memfr x k hd1 (by omega) (by omega), hs1mem]

set_option maxHeartbeats 0 in
/-- Trailing checks: cursor at buffer end, `main` below the
word count, and the main word's `ins` field zero; falls into
the checker entry. -/
theorem parser_trailing_ok {c e mn nw T : Nat} {s : X86Int.St}
    (hrip : s.rip = base + 0x419) (hr6 : s.reg 6 = some c)
    (hr15 : s.reg 15 = some e) (hr12 : s.reg 12 = some mn)
    (hr13 : s.reg 13 = some nw) (hr14 : s.reg 14 = some T)
    (hceq : c = e) (hcW : c < W) (heW : e < W)
    (hmn : mn < nw) (hTW : T + 64 * mn < W)
    (hslot : rd64 s.mem (T + 64 * mn) = some 0)
    (hlen16 : s.regs.length = 16) :
    ∃ s' : X86Int.St,
      run stage1Im 15 (Out.ok s) = Out.ok s' ∧
      s'.rip = base + 0x452 ∧ s'.reg 0 = some 0 ∧
      (∀ i, i ≠ 0 → i ≠ 2 → s'.reg i = s.reg i) ∧
      s'.mem = s.mem ∧ s'.regs.length = 16 ∧ s'.k = s.k := by
  have hc := stage1_parserCode
  have hlen : ∀ i : Nat, i < 16 → i < s.regs.length := by
    intro i hi; omega
  have hdbl : ∀ v : Nat, 2 * v < W → (aluF .add v v).1 = 2 * v := by
    intro v hv
    rw [aluF_add_fst, show v + v = 2 * v by omega,
      Nat.mod_eq_of_lt hv]
  -- 0x419: cmp rsi, r15
  have e1 : X86Int.step stage1Im s =
      .ok { s with fl := (aluF .sub c e).2, rip := s.rip + 3 } :=
    step_cmp (by rw [hrip]; exact hc.p419) hr6 hr15
  obtain ⟨t1, ht1⟩ : ∃ t : X86Int.St, t =
      { s with fl := (aluF .sub c e).2, rip := s.rip + 3 } := ⟨_, rfl⟩
  have ht1rip : t1.rip = base + 0x41c := by
    rw [ht1]; show s.rip + 3 = base + 0x41c; rw [hrip]
  have ht1fl : X86Int.cond t1.fl 4 = some true := by
    have hzf : t1.fl.zf = some true := by
      rw [ht1]
      show (aluF .sub c e).2.zf = some true
      rw [aluF_zf, hceq]
      have hbeq : ((aluF .sub e e).1 == 0) = true := by
        rw [beq_iff_eq]; exact sub_fst_self e
      rw [hbeq]
    have hcc := cond_cc4 hzf
    simpa using hcc
  have ht1fr : ∀ i, t1.reg i = s.reg i := by
    intro i
    rw [ht1]
    show s.reg i = s.reg i
    exact rfl
  have ht1mem : t1.mem = s.mem := by rw [ht1]
  have ht1len : t1.regs.length = 16 := by rw [ht1]; exact hlen16
  have ht1k : t1.k = s.k := by rw [ht1]
  have r1 : run stage1Im 1 (Out.ok s) = Out.ok t1 := by
    rw [run_succ_ok stage1Im 0 s, e1, ← ht1, run_zero]
  -- 0x41c: je 0x423
  have e2 : X86Int.step stage1Im t1 =
      .ok { t1 with rip := base + 0x423 } := by
    have h := step_jcc (im := stage1Im) (s := t1)
      (by rw [ht1rip]; exact hc.p41c) ht1fl
    simpa using h
  obtain ⟨t2, ht2⟩ : ∃ t : X86Int.St, t =
      { t1 with rip := base + 0x423 } := ⟨_, rfl⟩
  have ht2rip : t2.rip = base + 0x423 := by rw [ht2]
  have ht2fr : ∀ i, t2.reg i = s.reg i := by
    intro i
    rw [ht2]
    show ({ t1 with rip := base + 0x423 }).reg i = s.reg i
    rw [reg_with_rip]; exact ht1fr i
  have ht2mem : t2.mem = s.mem := by
    rw [ht2]
    show ({ t1 with rip := base + 0x423 }).mem = s.mem
    rw [mem_with_rip]; exact ht1mem
  have ht2len : t2.regs.length = 16 := by
    rw [ht2]
    show ({ t1 with rip := base + 0x423 }).regs.length = 16
    rw [len_with_rip]; exact ht1len
  have ht2k : t2.k = s.k := by rw [ht2]; exact ht1k
  have r2 : run stage1Im 1 (Out.ok t1) = Out.ok t2 := by
    rw [run_succ_ok stage1Im 0 t1, e2, ← ht2, run_zero]
  -- 0x423: cmp r12, r13
  have ht2r12 : t2.reg 12 = some mn := by rw [ht2fr 12]; exact hr12
  have ht2r13 : t2.reg 13 = some nw := by rw [ht2fr 13]; exact hr13
  have e3 : X86Int.step stage1Im t2 =
      .ok { t2 with fl := (aluF .sub mn nw).2, rip := t2.rip + 3 } :=
    step_cmp (by rw [ht2rip]; exact hc.p423) ht2r12 ht2r13
  obtain ⟨t3, ht3⟩ : ∃ t : X86Int.St, t =
      { t2 with fl := (aluF .sub mn nw).2, rip := t2.rip + 3 } :=
    ⟨_, rfl⟩
  have ht3rip : t3.rip = base + 0x426 := by
    rw [ht3]; show t2.rip + 3 = base + 0x426; rw [ht2rip]
  have ht3fl : X86Int.cond t3.fl 2 = some true := by
    have hcf : t3.fl.cf = some true := by
      rw [ht3]
      show (aluF .sub mn nw).2.cf = some true
      exact aluF_sub_cf_true hmn
    have hcc := cond_cc2 hcf
    simpa using hcc
  have ht3fr : ∀ i, t3.reg i = s.reg i := by
    intro i
    rw [ht3]
    show t2.reg i = s.reg i
    exact ht2fr i
  have ht3mem : t3.mem = s.mem := by rw [ht3]; exact ht2mem
  have ht3len : t3.regs.length = 16 := by rw [ht3]; exact ht2len
  have ht3k : t3.k = s.k := by rw [ht3]; exact ht2k
  have r3 : run stage1Im 1 (Out.ok t2) = Out.ok t3 := by
    rw [run_succ_ok stage1Im 0 t2, e3, ← ht3, run_zero]
  -- 0x426: jb 0x42d
  have e4 : X86Int.step stage1Im t3 =
      .ok { t3 with rip := base + 0x42d } := by
    have h := step_jcc (im := stage1Im) (s := t3)
      (by rw [ht3rip]; exact hc.p426) ht3fl
    simpa using h
  obtain ⟨t4, ht4⟩ : ∃ t : X86Int.St, t =
      { t3 with rip := base + 0x42d } := ⟨_, rfl⟩
  have ht4rip : t4.rip = base + 0x42d := by rw [ht4]
  have ht4fr : ∀ i, t4.reg i = s.reg i := by
    intro i
    rw [ht4]
    show ({ t3 with rip := base + 0x42d }).reg i = s.reg i
    rw [reg_with_rip]; exact ht3fr i
  have ht4mem : t4.mem = s.mem := by
    rw [ht4]
    show ({ t3 with rip := base + 0x42d }).mem = s.mem
    rw [mem_with_rip]; exact ht3mem
  have ht4len : t4.regs.length = 16 := by
    rw [ht4]
    show ({ t3 with rip := base + 0x42d }).regs.length = 16
    rw [len_with_rip]; exact ht3len
  have ht4k : t4.k = s.k := by rw [ht4]; exact ht3k
  have r4 : run stage1Im 1 (Out.ok t3) = Out.ok t4 := by
    rw [run_succ_ok stage1Im 0 t3, e4, ← ht4, run_zero]
  -- 0x42d: mov rdx, r12; six doublings; add rdx, r14
  have ht4r12 : t4.reg 12 = some mn := by rw [ht4fr 12]; exact hr12
  have e5 : X86Int.step stage1Im t4 =
      .ok { t4.set 2 (some mn) with rip := t4.rip + 3 } :=
    step_mov (by rw [ht4rip]; exact hc.p42d) ht4r12
  obtain ⟨t5, ht5⟩ : ∃ t : X86Int.St, t =
      { t4.set 2 (some mn) with rip := t4.rip + 3 } := ⟨_, rfl⟩
  have ht5rip : t5.rip = base + 0x430 := by
    rw [ht5]; show t4.rip + 3 = base + 0x430; rw [ht4rip]
  have ht5r2 : t5.reg 2 = some mn := by
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).reg 2 = some mn
    rw [reg_with_rip, reg_set_self (by rw [ht4len]; decide)]
  have ht5fr : ∀ i, i ≠ 2 → t5.reg i = s.reg i := by
    intro i hi
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).reg i = s.reg i
    rw [reg_with_rip, reg_set_other hi]; exact ht4fr i
  have ht5mem : t5.mem = s.mem := by
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht4mem
  have ht5len : t5.regs.length = 16 := by
    rw [ht5]
    show ({ t4.set 2 (some mn) with rip := t4.rip + 3 }).regs.length = 16
    rw [len_with_rip, len_set]; exact ht4len
  have ht5k : t5.k = s.k := by rw [ht5]; exact ht4k
  have r5 : run stage1Im 1 (Out.ok t4) = Out.ok t5 := by
    rw [run_succ_ok stage1Im 0 t4, e5, ← ht5, run_zero]
  -- six doublings of rdx, as one chain of explicit steps
  have dbl_step : ∀ {u : X86Int.St} {a v : Nat},
      stage1Im a = some (.alu .add 2 2, 3) → u.rip = a →
      u.reg 2 = some v → 2 * v < W → u.regs.length = 16 →
      ∃ u', run stage1Im 1 (Out.ok u) = Out.ok u' ∧
        u'.rip = a + 3 ∧ u'.reg 2 = some (2 * v) ∧
        (∀ i, i ≠ 2 → u'.reg i = u.reg i) ∧ u'.mem = u.mem ∧
        u'.regs.length = 16 ∧ u'.k = u.k := by
    intro u a v hdec hurip hur2 hv hulen
    have e : X86Int.step stage1Im u =
        .ok { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } :=
      step_alu_add (by rw [hurip]; exact hdec) hur2 hur2
    obtain ⟨u1, hu1⟩ : ∃ t : X86Int.St, t =
        { u.set 2 (some (aluF .add v v).1) with
          fl := (aluF .add v v).2, rip := u.rip + 3 } := ⟨_, rfl⟩
    have rr : run stage1Im 1 (Out.ok u) = Out.ok u1 := by
      rw [run_succ_ok stage1Im 0 u, e, ← hu1, run_zero]
    refine ⟨u1, rr, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · rw [hu1]; show u.rip + 3 = a + 3; rw [hurip]
    · rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).reg 2 = some (2 * v)
      rw [reg_set_self (by rw [hulen]; decide), hdbl v hv]
    · intro i hi
      rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).reg i = u.reg i
      rw [reg_set_other hi]
    · rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).mem = u.mem
      rw [mem_set]
    · rw [hu1]
      show (u.set 2 (some (aluF .add v v).1)).regs.length = 16
      rw [len_set]; exact hulen
    · rw [hu1]; exact rfl
  obtain ⟨t6, r6, ht6rip, ht6r2, ht6fr, ht6mem, ht6len, ht6k⟩ :=
    dbl_step hc.p430 ht5rip ht5r2 (by omega) ht5len
  have ht6rip' : t6.rip = base + 0x433 := ht6rip
  obtain ⟨t7, r7, ht7rip, ht7r2, ht7fr, ht7mem, ht7len, ht7k⟩ :=
    dbl_step hc.p433 ht6rip' ht6r2 (by omega) ht6len
  have ht7rip' : t7.rip = base + 0x436 := ht7rip
  have ht7r2' : t7.reg 2 = some (4 * mn) := by
    have h := ht7r2
    rwa [show 2 * (2 * mn) = 4 * mn by omega] at h
  obtain ⟨t8, r8, ht8rip, ht8r2, ht8fr, ht8mem, ht8len, ht8k⟩ :=
    dbl_step hc.p436 ht7rip' ht7r2' (by omega) ht7len
  have ht8rip' : t8.rip = base + 0x439 := ht8rip
  have ht8r2' : t8.reg 2 = some (8 * mn) := by
    have h := ht8r2
    rwa [show 2 * (4 * mn) = 8 * mn by omega] at h
  obtain ⟨t9, r9, ht9rip, ht9r2, ht9fr, ht9mem, ht9len, ht9k⟩ :=
    dbl_step hc.p439 ht8rip' ht8r2' (by omega) ht8len
  have ht9rip' : t9.rip = base + 0x43c := ht9rip
  have ht9r2' : t9.reg 2 = some (16 * mn) := by
    have h := ht9r2
    rwa [show 2 * (8 * mn) = 16 * mn by omega] at h
  obtain ⟨t10, r10, ht10rip, ht10r2, ht10fr, ht10mem, ht10len,
      ht10k⟩ :=
    dbl_step hc.p43c ht9rip' ht9r2' (by omega) ht9len
  have ht10rip' : t10.rip = base + 0x43f := ht10rip
  have ht10r2' : t10.reg 2 = some (32 * mn) := by
    have h := ht10r2
    rwa [show 2 * (16 * mn) = 32 * mn by omega] at h
  obtain ⟨t11, r11, ht11rip, ht11r2, ht11fr, ht11mem, ht11len,
      ht11k⟩ :=
    dbl_step hc.p43f ht10rip' ht10r2' (by omega) ht10len
  have ht11rip' : t11.rip = base + 0x442 := ht11rip
  have ht11r2' : t11.reg 2 = some (64 * mn) := by
    have h := ht11r2
    rwa [show 2 * (32 * mn) = 64 * mn by omega] at h
  have ht11fr' : ∀ i, i ≠ 2 → t11.reg i = s.reg i := by
    intro i hi
    rw [ht11fr i hi, ht10fr i hi, ht9fr i hi, ht8fr i hi, ht7fr i hi,
      ht6fr i hi]
    exact ht5fr i hi
  have ht11r14 : t11.reg 14 = some T := by
    rw [ht11fr' 14 (by decide)]; exact hr14
  have ht11mem' : t11.mem = s.mem := by
    rw [ht11mem, ht10mem, ht9mem, ht8mem, ht7mem, ht6mem]; exact ht5mem
  have ht11k' : t11.k = s.k := by
    rw [ht11k, ht10k, ht9k, ht8k, ht7k, ht6k]; exact ht5k
  -- 0x442: add rdx, r14
  have e12 : X86Int.step stage1Im t11 =
      .ok { t11.set 2 (some (aluF .add (64 * mn) T).1) with
        fl := (aluF .add (64 * mn) T).2, rip := t11.rip + 3 } :=
    step_alu_add (by rw [ht11rip']; exact hc.p442) ht11r2' ht11r14
  obtain ⟨t12, ht12⟩ : ∃ t : X86Int.St, t =
      { t11.set 2 (some (aluF .add (64 * mn) T).1) with
        fl := (aluF .add (64 * mn) T).2, rip := t11.rip + 3 } :=
    ⟨_, rfl⟩
  have ht12rip : t12.rip = base + 0x445 := by
    rw [ht12]; show t11.rip + 3 = base + 0x445; rw [ht11rip']
  have ht12r2 : t12.reg 2 = some (T + 64 * mn) := by
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).reg 2 =
      some (T + 64 * mn)
    rw [reg_set_self (by rw [ht11len]; decide), aluF_add_fst,
      show 64 * mn + T = T + 64 * mn by omega,
      Nat.mod_eq_of_lt hTW]
  have ht12fr : ∀ i, i ≠ 2 → t12.reg i = s.reg i := by
    intro i hi
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).reg i = s.reg i
    rw [reg_set_other hi]; exact ht11fr' i hi
  have ht12mem : t12.mem = s.mem := by
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).mem = s.mem
    rw [mem_set]; exact ht11mem'
  have ht12len : t12.regs.length = 16 := by
    rw [ht12]
    show (t11.set 2 (some (aluF .add (64 * mn) T).1)).regs.length = 16
    rw [len_set]; exact ht11len
  have ht12k : t12.k = s.k := by rw [ht12]; exact ht11k'
  have r12 : run stage1Im 1 (Out.ok t11) = Out.ok t12 := by
    rw [run_succ_ok stage1Im 0 t11, e12, ← ht12, run_zero]
  -- 0x445: ld rax, [rdx]
  have hld : rd64 t12.mem (ea (T + 64 * mn) 0) = some 0 := by
    unfold ea
    rw [Nat.add_zero, Nat.mod_eq_of_lt hTW, ht12mem]; exact hslot
  have e13 : X86Int.step stage1Im t12 =
      .ok { t12.set 0 (some 0) with rip := t12.rip + 3 } :=
    step_ld (by rw [ht12rip]; exact hc.p445) ht12r2 hld
  obtain ⟨t13, ht13⟩ : ∃ t : X86Int.St, t =
      { t12.set 0 (some 0) with rip := t12.rip + 3 } := ⟨_, rfl⟩
  have ht13rip : t13.rip = base + 0x448 := by
    rw [ht13]; show t12.rip + 3 = base + 0x448; rw [ht12rip]
  have ht13r0 : t13.reg 0 = some 0 := by
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).reg 0 = some 0
    rw [reg_with_rip, reg_set_self (by rw [ht12len]; decide)]
  have ht13fr : ∀ i, i ≠ 0 → i ≠ 2 → t13.reg i = s.reg i := by
    intro i hi0 hi2
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).reg i =
      s.reg i
    rw [reg_with_rip, reg_set_other hi0]; exact ht12fr i hi2
  have ht13mem : t13.mem = s.mem := by
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).mem = s.mem
    rw [mem_with_rip, mem_set]; exact ht12mem
  have ht13len : t13.regs.length = 16 := by
    rw [ht13]
    show ({ t12.set 0 (some 0) with rip := t12.rip + 3 }).regs.length =
      16
    rw [len_with_rip, len_set]; exact ht12len
  have ht13k : t13.k = s.k := by rw [ht13]; exact ht12k
  have r13 : run stage1Im 1 (Out.ok t12) = Out.ok t13 := by
    rw [run_succ_ok stage1Im 0 t12, e13, ← ht13, run_zero]
  -- 0x448: test rax, rax; 0x44b: je checker entry
  have e14 : X86Int.step stage1Im t13 =
      .ok { t13 with fl := (aluF .and 0 0).2, rip := t13.rip + 3 } :=
    step_test (by rw [ht13rip]; exact hc.p448) ht13r0 ht13r0
  obtain ⟨t14, ht14⟩ : ∃ t : X86Int.St, t =
      { t13 with fl := (aluF .and 0 0).2, rip := t13.rip + 3 } :=
    ⟨_, rfl⟩
  have ht14rip : t14.rip = base + 0x44b := by
    rw [ht14]; show t13.rip + 3 = base + 0x44b; rw [ht13rip]
  have ht14fl : X86Int.cond t14.fl 4 = some true := by
    have hzf : t14.fl.zf = some true := by
      rw [ht14]
      show (aluF .and 0 0).2.zf = some true
      rw [aluF_zf, aluF_and_fst_self]
      simp
    have hcc := cond_cc4 hzf
    simpa using hcc
  have r14 : run stage1Im 1 (Out.ok t13) = Out.ok t14 := by
    rw [run_succ_ok stage1Im 0 t13, e14, ← ht14, run_zero]
  have e15 : X86Int.step stage1Im t14 =
      .ok { t14 with rip := base + 0x452 } := by
    have h := step_jcc (im := stage1Im) (s := t14)
      (by rw [ht14rip]; exact hc.p44b) ht14fl
    simpa using h
  obtain ⟨t15, ht15⟩ : ∃ t : X86Int.St, t =
      { t14 with rip := base + 0x452 } := ⟨_, rfl⟩
  have r15 : run stage1Im 1 (Out.ok t14) = Out.ok t15 := by
    rw [run_succ_ok stage1Im 0 t14, e15, ← ht15, run_zero]
  have hrun : run stage1Im 15 (Out.ok s) = Out.ok t15 := by
    rw [show (15 : Nat) = 1 + 14 by omega, run_add stage1Im 1 14, r1,
      show (14 : Nat) = 1 + 13 by omega, run_add stage1Im 1 13, r2,
      show (13 : Nat) = 1 + 12 by omega, run_add stage1Im 1 12, r3,
      show (12 : Nat) = 1 + 11 by omega, run_add stage1Im 1 11, r4,
      show (11 : Nat) = 1 + 10 by omega, run_add stage1Im 1 10, r5,
      show (10 : Nat) = 1 + 9 by omega, run_add stage1Im 1 9, r6,
      show (9 : Nat) = 1 + 8 by omega, run_add stage1Im 1 8, r7,
      show (8 : Nat) = 1 + 7 by omega, run_add stage1Im 1 7, r8,
      show (7 : Nat) = 1 + 6 by omega, run_add stage1Im 1 6, r9,
      show (6 : Nat) = 1 + 5 by omega, run_add stage1Im 1 5, r10,
      show (5 : Nat) = 1 + 4 by omega, run_add stage1Im 1 4, r11,
      show (4 : Nat) = 1 + 3 by omega, run_add stage1Im 1 3, r12,
      show (3 : Nat) = 1 + 2 by omega, run_add stage1Im 1 2, r13,
      show (2 : Nat) = 1 + 1 by omega, run_add stage1Im 1 1, r14, r15]
  refine ⟨t15, hrun, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · rw [ht15]
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).reg 0 = some 0
    rw [reg_with_rip, ht14]
    show t13.reg 0 = some 0
    exact ht13r0
  · intro i hi0 hi2
    rw [ht15]
    show ({ t14 with rip := base + 0x452 }).reg i = s.reg i
    rw [reg_with_rip, ht14]
    show t13.reg i = s.reg i
    exact ht13fr i hi0 hi2
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).mem = s.mem
    rw [mem_with_rip, ht14]
    show t13.mem = s.mem
    exact ht13mem
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).regs.length = 16
    rw [len_with_rip, ht14]
    show t13.regs.length = 16
    exact ht13len
  · rw [ht15]
    show ({ t14 with rip := base + 0x452 }).k = s.k
    rw [ht14]
    show t13.k = s.k
    exact ht13k

/-- `leBytes` has the requested length. -/
theorem leBytes_length (n k : Nat) : (leBytes n k).length = k := by
  simp [leBytes]

/-- Head/tail decomposition of `leBytes`. -/
theorem leBytes_succ (n k : Nat) :
    leBytes n (k + 1) = (n % 256) :: leBytes (n / 256) k := by
  apply List.ext_getElem
  · simp [leBytes]
  · intro i hi1 _
    cases i with
    | zero => simp [leBytes]
    | succ j =>
      have hj : j < k := by
        have h := hi1
        simp [leBytes] at h
        omega
      have e1 : (leBytes n (k + 1))[j + 1]'(by
            simpa [leBytes] using hi1) = n / 256 ^ (j + 1) % 256 := by
        simp [leBytes, hj]
      have e2 : ((n % 256) :: leBytes (n / 256) k)[j + 1]'(by
            simp [leBytes, hj]) = n / 256 / 256 ^ j % 256 := by
        simp [leBytes, hj]
      rw [e1, e2, Nat.div_div_eq_div_mul]
      congr 1
      rw [Nat.pow_succ']

/-- A byte-bounded list has `leVal` below the place bound. -/
theorem leVal_bound : ∀ {bs : List Nat}, (∀ b ∈ bs, b < 256) →
    leVal bs < 256 ^ bs.length
  | [], _ => by decide
  | b :: bs, h => by
    have hb : b < 256 := h b (.head _)
    have ih := leVal_bound (fun x hx => h x (.tail _ hx))
    show leVal (b :: bs) < 256 ^ (b :: bs).length
    simp only [leVal, List.length_cons, Nat.pow_succ]
    omega

/-- `leBytes` of a byte-bounded list's `leVal` is the list. -/
theorem leBytes_leVal : ∀ {bs : List Nat}, (∀ b ∈ bs, b < 256) →
    leBytes (leVal bs) bs.length = bs
  | [], _ => by simp [leBytes, leVal]
  | b :: bs, h => by
    have hb : b < 256 := h b (.head _)
    have ih := leBytes_leVal (fun x hx => h x (.tail _ hx))
    show leBytes (leVal (b :: bs)) (b :: bs).length = b :: bs
    simp only [List.length_cons]
    rw [show leVal (b :: bs) = b + 256 * leVal bs from rfl,
      leBytes_succ]
    congr 1
    · rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hb]
    · have e : (b + 256 * leVal bs) / 256 = leVal bs := by
        rw [Nat.add_mul_div_left _ _ (by decide : 0 < 256),
          Nat.div_eq_of_lt hb, Nat.zero_add]
      rw [e]; exact ih

/-- The `rdLE` fold is `leVal`, at a shifted range. -/
theorem fold_zip_range'_aux : ∀ (l : List Nat) (s acc : Nat),
    (l.zip (List.range' s l.length)).foldl
      (fun acc x => acc + x.1 * 256 ^ x.2) acc =
      acc + 256 ^ s * leVal l
  | [], _, _ => by
    rw [show leVal [] = 0 from rfl]; simp
  | b :: bs, s, acc => by
    have ih := fold_zip_range'_aux bs (s + 1) (acc + b * 256 ^ s)
    show (((b :: bs).zip
      (List.range' s (b :: bs).length)).foldl _ acc) = _
    simp only [List.length_cons, List.range'_succ, List.zip_cons_cons,
      List.foldl_cons]
    rw [ih]
    have e1 : 256 ^ (s + 1) * leVal bs = 256 ^ s * (256 * leVal bs) := by
      rw [Nat.pow_succ', Nat.mul_assoc, Nat.mul_left_comm (256 ^ s) 256]
    have e2 : leVal (b :: bs) = b + 256 * leVal bs := rfl
    rw [e1, e2, Nat.mul_add, Nat.mul_comm b (256 ^ s)]
    omega

/-- The `rdLE` fold is `leVal`. -/
theorem fold_zip_range (l : List Nat) :
    (l.zip (List.range l.length)).foldl
      (fun acc x => acc + x.1 * 256 ^ x.2) 0 = leVal l := by
  rw [List.range_eq_range']
  have h := fold_zip_range'_aux l 0 0
  simpa using h

/-- What a successful `rdLE` says. -/
theorem rdLE_eq {bs : List Nat} {k v : Nat} {rest : List Nat}
    (h : rdLE bs k = some (v, rest)) :
    v = leVal (bs.take k) ∧ rest = bs.drop k ∧ k ≤ bs.length := by
  unfold rdLE at h
  by_cases hlt : bs.length < k
  · rw [if_pos hlt] at h
    exact absurd h (by simp)
  · rw [if_neg hlt] at h
    obtain ⟨hv, hr⟩ := Prod.mk.inj (Option.some.inj h)
    refine ⟨?_, hr.symm, by omega⟩
    have hlen : (bs.take k).length = k := by
      rw [List.length_take]
      omega
    have h2 := fold_zip_range (bs.take k)
    rw [hlen] at h2
    rw [← h2]
    exact hv.symm

/-- A successful `rdLE` on bounded bytes re-encodes exactly. -/
theorem rdLE_inv {bs : List Nat} {k v : Nat} {rest : List Nat}
    (h : rdLE bs k = some (v, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = leBytes v k ++ rest ∧ v < 256 ^ k ∧ k ≤ bs.length := by
  obtain ⟨hv, hr, hk⟩ := rdLE_eq h
  have hlen : (bs.take k).length = k := by
    rw [List.length_take]; omega
  have hmem : ∀ b ∈ bs.take k, b < 256 := fun b hbmem =>
    hb b (List.mem_of_mem_take hbmem)
  have htake : leBytes v k = bs.take k := by
    have ht := leBytes_leVal hmem
    rw [hlen, ← hv] at ht
    exact ht
  have hbound : v < 256 ^ k := by
    have ht := leVal_bound hmem
    rw [hlen] at ht
    rw [hv]
    exact ht
  refine ⟨?_, hbound, hk⟩
  calc bs = bs.take k ++ bs.drop k := (List.take_append_drop k bs).symm
    _ = leBytes v k ++ rest := by rw [← htake, ← hr]

/-- Immediate bounds forced by the fixed-width encodings. -/
def InstrBound : Instr → Prop
  | .prim _ => True
  | .lit n => n < 256 ^ 8
  | .str i => i < 256 ^ 4
  | .pick n => n < 256 ^ 4
  | .tor => True
  | .fromr => True
  | .rfetch => True
  | .locget i => i < 256 ^ 4
  | .locset i => i < 256 ^ 4
  | .jmp t => t < 256 ^ 4
  | .jz t => t < 256 ^ 4
  | .jnz t => t < 256 ^ 4
  | .call w => w < 256 ^ 4
  | .ret => True

/-- There are 57 primitives. -/
theorem prim_all_length : Prim.all.length = 57 := by decide

/-- The code of the `op`-th primitive is `op`. -/
theorem prim_code_get (op : Nat) (h : op < Prim.all.length) :
    (Prim.all[op]'h).code = op := by
  have h57 : op < 57 := by
    rw [prim_all_length] at h; exact h
  have e : Prim.all[op]? = some (Prim.all[op]'h) :=
    List.getElem?_eq_getElem h
  have chk : (List.range 57).all
      (fun i => (Prim.all[i]?.map Prim.code) == some i) = true := by decide
  have hm := List.all_eq_true.mp chk op (List.mem_range.mpr h57)
  rw [e] at hm
  simp at hm
  exact hm

/-- An opcode that is not a primitive and not a known opcode
decodes to nothing. -/
theorem decInstr_none_of {op : Nat} {r : List Nat}
    (hop : ¬ op < Prim.all.length)
    (h80 : op ≠ 0x80) (h81 : op ≠ 0x81) (h82 : op ≠ 0x82)
    (h83 : op ≠ 0x83) (h84 : op ≠ 0x84) (h85 : op ≠ 0x85)
    (h86 : op ≠ 0x86) (h87 : op ≠ 0x87) (h88 : op ≠ 0x88)
    (h89 : op ≠ 0x89) (h8a : op ≠ 0x8a) (h8b : op ≠ 0x8b)
    (h8c : op ≠ 0x8c) :
    decInstr (op :: r) = none := by
  unfold decInstr
  split
  · simp_all
  · split
    · simp_all <;> omega
    · split <;> simp_all

/-- Inversion of an immediate-carrying decode. -/
theorem imm_inv {k op : Nat} {f : Nat → Instr}
    {r rest : List Nat} {x : Instr}
    (hb : ∀ b ∈ r, b < 256)
    (h : (rdLE r k).map (fun y => (f y.1, y.2)) = some (x, rest))
    (henc : ∀ v, (f v).enc = op :: leBytes v k) :
    ∃ v, x = f v ∧ v < 256 ^ k ∧ op :: r = x.enc ++ rest := by
  rw [Option.map_eq_some_iff] at h
  obtain ⟨⟨v, r2⟩, hr, hfx⟩ := h
  have hx : f v = x := congrArg Prod.fst hfx
  have hr2 : r2 = rest := congrArg Prod.snd hfx
  obtain ⟨hr0, hbnd, -⟩ := rdLE_inv hr hb
  refine ⟨v, hx.symm, hbnd, ?_⟩
  calc op :: r = op :: (leBytes v k ++ r2) := by rw [hr0]
    _ = (op :: leBytes v k) ++ rest := by
      simp only [List.cons_append, hr2]
    _ = x.enc ++ rest := by rw [← henc v, hx]

/-- A decoded instruction re-encodes to exactly the bytes it
consumed, with bounded immediates. -/
theorem decInstr_inv {bs rest : List Nat} {x : Instr}
    (h : decInstr bs = some (x, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = x.enc ++ rest ∧ InstrBound x := by
  cases bs with
  | nil => simp [decInstr] at h
  | cons op rest0 =>
    have hb0 : ∀ b ∈ rest0, b < 256 := fun b hbmem =>
      hb b (.tail _ hbmem)
    by_cases hop : op < Prim.all.length
    · have e : decInstr (op :: rest0) =
          some (Instr.prim (Prim.all[op]'hop), rest0) := by
        unfold decInstr
        split
        · simp_all
        · split
          · simp_all
          · simp_all <;> omega
      rw [e] at h
      obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
      subst hx; subst hr
      refine ⟨?_, trivial⟩
      show op :: rest0 = (Instr.prim (Prim.all[op]'hop)).enc ++ rest0
      show op :: rest0 = [(Prim.all[op]'hop).code] ++ rest0
      simp only [prim_code_get op hop, List.singleton_append]
    · have hcases : op = 0x80 ∨ op = 0x81 ∨ op = 0x82 ∨ op = 0x83 ∨
          op = 0x84 ∨ op = 0x85 ∨ op = 0x86 ∨ op = 0x87 ∨ op = 0x88 ∨
          op = 0x89 ∨ op = 0x8a ∨ op = 0x8b ∨ op = 0x8c ∨
          (op ≠ 0x80 ∧ op ≠ 0x81 ∧ op ≠ 0x82 ∧ op ≠ 0x83 ∧ op ≠ 0x84 ∧
          op ≠ 0x85 ∧ op ≠ 0x86 ∧ op ≠ 0x87 ∧ op ≠ 0x88 ∧ op ≠ 0x89 ∧
          op ≠ 0x8a ∧ op ≠ 0x8b ∧ op ≠ 0x8c) := by omega
      rcases hcases with rfl | rfl | rfl | rfl | rfl | rfl | rfl |
        rfl | rfl | rfl | rfl | rfl | rfl |
        ⟨h80, h81, h82, h83, h84, h85, h86, h87, h88, h89, h8a, h8b,
          h8c⟩
      · have e : decInstr (0x80 :: rest0) =
            (rdLE rest0 8).map (fun y => (.lit y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x81 :: rest0) =
            (rdLE rest0 4).map (fun y => (.str y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x82 :: rest0) =
            (rdLE rest0 4).map (fun y => (.pick y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x83 :: rest0) = some (.tor, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · have e : decInstr (0x84 :: rest0) = some (.fromr, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · have e : decInstr (0x85 :: rest0) = some (.rfetch, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · have e : decInstr (0x86 :: rest0) =
            (rdLE rest0 4).map (fun y => (.locget y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x87 :: rest0) =
            (rdLE rest0 4).map (fun y => (.locset y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x88 :: rest0) =
            (rdLE rest0 4).map (fun y => (.jmp y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x89 :: rest0) =
            (rdLE rest0 4).map (fun y => (.jz y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x8a :: rest0) =
            (rdLE rest0 4).map (fun y => (.jnz y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x8b :: rest0) =
            (rdLE rest0 4).map (fun y => (.call y.1, y.2)) := rfl
        rw [e] at h
        obtain ⟨v, hx, hbnd, hbs⟩ :=
          imm_inv hb0 h (fun v => rfl)
        exact ⟨hbs, hx.symm ▸ hbnd⟩
      · have e : decInstr (0x8c :: rest0) = some (.ret, rest0) := rfl
        rw [e] at h
        obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
        subst hx; subst hr
        exact ⟨rfl, trivial⟩
      · rw [decInstr_none_of hop h80 h81 h82 h83 h84 h85 h86 h87 h88
          h89 h8a h8b h8c] at h
        simp at h

/-- Inversion of `decMany`: the consumed bytes are the flat
encodings of the items, the count matches, and every item
satisfies the per-item predicate. -/
theorem decMany_inv {α : Type} {f : List Nat → Option (α × List Nat)}
    {e : α → List Nat} {P : α → Prop}
    (hf : ∀ {bs rest} {x : α}, f bs = some (x, rest) →
      (∀ b ∈ bs, b < 256) → bs = e x ++ rest ∧ P x) :
    ∀ {n bs rest} {xs : List α}, decMany f n bs = some (xs, rest) →
      (∀ b ∈ bs, b < 256) →
      bs = (xs.flatMap e) ++ rest ∧ xs.length = n ∧ ∀ x ∈ xs, P x := by
  intro n
  induction n with
  | zero =>
    intro bs rest xs h hb
    rw [show decMany f 0 bs = some ([], bs) from rfl] at h
    obtain ⟨hx, hr⟩ := Prod.mk.inj (Option.some.inj h)
    subst hx; subst hr
    exact ⟨rfl, rfl, fun x hx => by simp at hx⟩
  | succ n ih =>
    intro bs rest xs h hb
    have hstep : decMany f (n + 1) bs =
        (f bs).bind (fun p =>
          (decMany f n p.2).bind (fun q => some (p.1 :: q.1, q.2))) := rfl
    rw [hstep, Option.bind_eq_some_iff] at h
    obtain ⟨⟨x, bs1⟩, h1, h2⟩ := h
    rw [Option.bind_eq_some_iff] at h2
    obtain ⟨⟨xs', bs2⟩, h3, h4⟩ := h2
    have h3' : decMany f n bs1 = some (xs', bs2) := h3
    have hx : x :: xs' = xs :=
      congrArg Prod.fst (Option.some.inj h4)
    have hr : bs2 = rest :=
      congrArg Prod.snd (Option.some.inj h4)
    subst hx; subst hr
    obtain ⟨he1, hPx⟩ := hf h1 hb
    have hb1 : ∀ b ∈ bs1, b < 256 := fun b hbmem =>
      hb b (he1.symm ▸ List.mem_append_right _ hbmem)
    obtain ⟨he2, hlen, hP⟩ := ih h3' hb1
    refine ⟨?_, by simp [hlen], ?_⟩
    · calc bs = e x ++ bs1 := he1
        _ = e x ++ ((xs'.flatMap e) ++ bs2) := by rw [he2]
        _ = ((x :: xs').flatMap e) ++ bs2 := by
          rw [List.flatMap_cons, List.append_assoc]
    · intro z hz
      simp only [List.mem_cons] at hz
      rcases hz with rfl | hz
      · exact hPx
      · exact hP z hz

/-- The per-annotation byte bounds. -/
def AnnBound (x : Nat × Height) : Prop :=
  x.1 < 256 ^ 4 ∧ x.2.d < 256 ^ 4 ∧ x.2.a < 256 ^ 4

/-- Inversion of `decAnn`. -/
theorem decAnn_inv {bs rest : List Nat} {x : Nat × Height}
    (h : decAnn bs = some (x, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = (leBytes x.1 4 ++ leBytes x.2.d 4 ++ leBytes x.2.a 4) ++
      rest ∧ AnnBound x := by
  have hstep : decAnn bs =
      (rdLE bs 4).bind (fun p =>
        (rdLE p.2 4).bind (fun q =>
          (rdLE q.2 4).bind
            (fun r => some ((p.1, ⟨q.1, r.1⟩), r.2)))) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨i, bs1⟩, h1, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨d, bs2⟩, h2, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨a, bs3⟩, h3, h4⟩ := h
  have h2' : rdLE bs1 4 = some (d, bs2) := h2
  have h3' : rdLE bs2 4 = some (a, bs3) := h3
  have hx : (i, (⟨d, a⟩ : Height)) = x :=
    congrArg Prod.fst (Option.some.inj h4)
  have hr : bs3 = rest :=
    congrArg Prod.snd (Option.some.inj h4)
  subst hr
  obtain ⟨he1, hbi, -⟩ := rdLE_inv h1 hb
  have hb1 : ∀ b ∈ bs1, b < 256 := fun b hbmem =>
    hb b (he1.symm ▸ List.mem_append_right _ hbmem)
  obtain ⟨he2, hbd, -⟩ := rdLE_inv h2' hb1
  have hb2 : ∀ b ∈ bs2, b < 256 := fun b hbmem =>
    hb1 b (he2.symm ▸ List.mem_append_right _ hbmem)
  obtain ⟨he3, hba, -⟩ := rdLE_inv h3' hb2
  subst hx
  refine ⟨?_, hbi, hbd, hba⟩
  calc bs = leBytes i 4 ++ bs1 := he1
    _ = leBytes i 4 ++ (leBytes d 4 ++ bs2) := by rw [he2]
    _ = leBytes i 4 ++ (leBytes d 4 ++ (leBytes a 4 ++ bs3)) := by
      rw [he3]
    _ = (leBytes i 4 ++ leBytes d 4 ++ leBytes a 4) ++ bs3 := by
      simp only [List.append_assoc]

/-- Inversion of `decData`. -/
theorem decData_inv {bs rest d : List Nat}
    (h : decData bs = some (d, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = (leBytes d.length 4 ++ d) ++ rest ∧ d.length < 256 ^ 4 := by
  have hstep : decData bs =
      (rdLE bs 4).bind (fun p =>
        if p.2.length < p.1 then none
        else some (p.2.take p.1, p.2.drop p.1)) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨n, bs1⟩, h1, h⟩ := h
  by_cases hlt : bs1.length < n
  · rw [if_pos hlt] at h
    exact absurd h (by simp)
  · rw [if_neg hlt] at h
    have hd : bs1.take n = d := congrArg Prod.fst (Option.some.inj h)
    have hr : bs1.drop n = rest := congrArg Prod.snd (Option.some.inj h)
    obtain ⟨he1, hbn, -⟩ := rdLE_inv h1 hb
    have hdlen : d.length = n := by
      rw [← hd, List.length_take]
      omega
    refine ⟨?_, by rw [hdlen]; exact hbn⟩
    calc bs = leBytes n 4 ++ bs1 := he1
      _ = leBytes n 4 ++ (bs1.take n ++ bs1.drop n) := by
        rw [List.take_append_drop]
      _ = (leBytes d.length 4 ++ d) ++ rest := by
        rw [hdlen, ← hd, ← hr]
        simp only [List.append_assoc]

/-- The per-word byte bounds. -/
def WordBound (w : Word) : Prop :=
  w.ins < 256 ^ 4 ∧ w.outs < 256 ^ 4 ∧ w.locals < 256 ^ 4 ∧
    w.code.length < 256 ^ 4 ∧ w.ann.length < 256 ^ 4 ∧
    (∀ x ∈ w.code, InstrBound x) ∧ (∀ x ∈ w.ann, AnnBound x)

/-- Inversion of `decWord`. -/
theorem decWord_inv {bs rest : List Nat} {w : Word}
    (h : decWord bs = some (w, rest)) (hb : ∀ b ∈ bs, b < 256) :
    bs = w.enc ++ rest ∧ WordBound w := by
  have hstep : decWord bs =
      (rdLE bs 4).bind (fun p1 =>
        (rdLE p1.2 4).bind (fun p2 =>
          (rdLE p2.2 4).bind (fun p3 =>
            (rdLE p3.2 4).bind (fun p4 =>
              (rdLE p4.2 4).bind (fun p5 =>
                (decMany decInstr p4.1 p5.2).bind (fun p6 =>
                  (decMany decAnn p5.1 p6.2).bind (fun p7 =>
                    some (⟨p1.1, p2.1, p3.1, p6.1, p7.1⟩,
                      p7.2)))))))) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨vi, bs1⟩, h1, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨vo, bs2⟩, h2, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨vl, bs3⟩, h3, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨nc, bs4⟩, h4, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨na, bs5⟩, h5, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨code, bs6⟩, h6, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨ann, bs7⟩, h7, h8⟩ := h
  have h2' : rdLE bs1 4 = some (vo, bs2) := h2
  have h3' : rdLE bs2 4 = some (vl, bs3) := h3
  have h4' : rdLE bs3 4 = some (nc, bs4) := h4
  have h5' : rdLE bs4 4 = some (na, bs5) := h5
  have h6' : decMany decInstr nc bs5 = some (code, bs6) := h6
  have h7' : decMany decAnn na bs6 = some (ann, bs7) := h7
  have hw : (⟨vi, vo, vl, code, ann⟩ : Word) = w :=
    congrArg Prod.fst (Option.some.inj h8)
  have hr : bs7 = rest := congrArg Prod.snd (Option.some.inj h8)
  subst hr; subst hw
  obtain ⟨he1, hbi, -⟩ := rdLE_inv h1 hb
  have hb1 : ∀ b ∈ bs1, b < 256 := fun b hm =>
    hb b (he1.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he2, hbo, -⟩ := rdLE_inv h2' hb1
  have hb2 : ∀ b ∈ bs2, b < 256 := fun b hm =>
    hb1 b (he2.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he3, hbl, -⟩ := rdLE_inv h3' hb2
  have hb3 : ∀ b ∈ bs3, b < 256 := fun b hm =>
    hb2 b (he3.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he4, hbnc, -⟩ := rdLE_inv h4' hb3
  have hb4 : ∀ b ∈ bs4, b < 256 := fun b hm =>
    hb3 b (he4.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he5, hbna, -⟩ := rdLE_inv h5' hb4
  have hb5 : ∀ b ∈ bs5, b < 256 := fun b hm =>
    hb4 b (he5.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he6, hclen, hcP⟩ :=
    decMany_inv (f := decInstr) (e := Instr.enc) (P := InstrBound)
      (fun hh hb' => decInstr_inv hh hb') h6' hb5
  have hb6 : ∀ b ∈ bs6, b < 256 := fun b hm =>
    hb5 b (he6.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he7, halen, haP⟩ :=
    decMany_inv (f := decAnn)
      (e := fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4) (P := AnnBound)
      (fun hh hb' => decAnn_inv hh hb') h7' hb6
  have hwenc : (⟨vi, vo, vl, code, ann⟩ : Word).enc =
      leBytes vi 4 ++ leBytes vo 4 ++ leBytes vl 4 ++
        leBytes code.length 4 ++ leBytes ann.length 4 ++
        (code.flatMap Instr.enc) ++
        (ann.flatMap fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
          leBytes x.2.a 4) := rfl
  refine ⟨?_, hbi, hbo, hbl,
    by show code.length < 256 ^ 4; rw [hclen]; exact hbnc,
    by show ann.length < 256 ^ 4; rw [halen]; exact hbna, hcP, haP⟩
  rw [hwenc, he1, he2, he3, he4, he5, he6, he7, hclen, halen]
  simp only [List.append_assoc]

/-- Inversion of `decode`: a decoded file is exactly the encoding
of its program, all header, word, code, annotation and data
fields being byte-bounded as their widths force. -/
theorem decode_inv {f : List Nat} {pr : Prog}
    (h : decode f = some pr) (hb : ∀ b ∈ f, b < 256) :
    f = pr.enc ∧ pr.main < 256 ^ 4 ∧ (∀ w ∈ pr.words, WordBound w) ∧
      (∀ d ∈ pr.data, d.length < 256 ^ 4) := by
  have hmagic : ∃ t, f = 0x53 :: 0x50 :: 0x52 :: 0x31 :: t := by
    unfold Sporos.decode at h
    split at h
    · exact ⟨_, rfl⟩
    · simp at h
  obtain ⟨t, rfl⟩ := hmagic
  have hbt : ∀ b ∈ t, b < 256 := fun b hm =>
    hb b (by simp [hm])
  have hstep : Sporos.decode (0x53 :: 0x50 :: 0x52 :: 0x31 :: t) =
      (rdLE t 4).bind (fun p1 =>
        (rdLE p1.2 4).bind (fun p2 =>
          (rdLE p2.2 4).bind (fun p3 =>
            (decMany decData p3.1 p3.2).bind (fun p4 =>
              (decMany decWord p2.1 p4.2).bind (fun p5 =>
                if p5.2.isEmpty = true then
                  some (⟨p5.1, p4.1, p1.1⟩ : Prog)
                else none))))) := rfl
  rw [hstep, Option.bind_eq_some_iff] at h
  obtain ⟨⟨mn, t1⟩, h1, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨nw, t2⟩, h2, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨nd, t3⟩, h3, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨data, t4⟩, h4, h⟩ := h
  rw [Option.bind_eq_some_iff] at h
  obtain ⟨⟨words, t5⟩, h5, h6⟩ := h
  have h2' : rdLE t1 4 = some (nw, t2) := h2
  have h3' : rdLE t2 4 = some (nd, t3) := h3
  have h4' : decMany decData nd t3 = some (data, t4) := h4
  have h5' : decMany decWord nw t4 = some (words, t5) := h5
  have h6' : (if t5.isEmpty = true then
      some (⟨words, data, mn⟩ : Prog) else none) = some pr := h6
  have ht5e : t5.isEmpty = true := by
    cases hbe : t5.isEmpty with
    | true => rfl
    | false =>
      rw [if_neg (by simp [hbe])] at h6'
      exact absurd h6' (by simp)
  rw [if_pos ht5e] at h6'
  have hpr : (⟨words, data, mn⟩ : Prog) = pr := Option.some.inj h6'
  have ht5 : t5 = [] := by
    cases t5 with
    | nil => rfl
    | cons a l => simp at ht5e
  obtain ⟨he1, hbmn, -⟩ := rdLE_inv h1 hbt
  have hb1 : ∀ b ∈ t1, b < 256 := fun b hm =>
    hbt b (he1.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he2, -, -⟩ := rdLE_inv h2' hb1
  have hb2 : ∀ b ∈ t2, b < 256 := fun b hm =>
    hb1 b (he2.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he3, -, -⟩ := rdLE_inv h3' hb2
  have hb3 : ∀ b ∈ t3, b < 256 := fun b hm =>
    hb2 b (he3.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he4, hdlen, hdP⟩ :=
    decMany_inv (f := decData)
      (e := fun d => leBytes d.length 4 ++ d)
      (P := fun d => d.length < 256 ^ 4)
      (fun hh hb' => decData_inv hh hb') h4' hb3
  have hb4 : ∀ b ∈ t4, b < 256 := fun b hm =>
    hb3 b (he4.symm ▸ List.mem_append_right _ hm)
  obtain ⟨he5, hwlen, hwP⟩ :=
    decMany_inv (f := decWord) (e := Word.enc) (P := WordBound)
      (fun hh hb' => decWord_inv hh hb') h5' hb4
  have henc : (⟨words, data, mn⟩ : Prog).enc =
      [0x53, 0x50, 0x52, 0x31] ++ leBytes mn 4 ++
        leBytes words.length 4 ++ leBytes data.length 4 ++
        (data.flatMap fun d => leBytes d.length 4 ++ d) ++
        words.flatMap Word.enc := rfl
  subst hpr
  refine ⟨?_, hbmn, hwP, hdP⟩
  rw [henc, he1, he2, he3, he4, he5, ht5, hwlen, hdlen]
  simp only [List.append_assoc, List.append_nil, List.cons_append,
    List.nil_append]

/-- The value of a `u32` field below its bound, read at `W`. -/
theorem leVal4_W {v : Nat} (h : v < 256 ^ 4) :
    leVal (leBytes v 4) % W = v := by
  rw [leVal_leBytes, Nat.mod_eq_of_lt h,
    Nat.mod_eq_of_lt (Nat.lt_trans h (by unfold W; decide))]

/-- The value of a `u32` field, read at `2^32`. -/
theorem leVal4_32 (v : Nat) :
    leVal (leBytes v 4) % 2 ^ 32 = v % 2 ^ 32 := by
  rw [leVal_leBytes]
  show v % 2 ^ 32 % 2 ^ 32 = v % 2 ^ 32
  exact Nat.mod_mod v (2 ^ 32)

/-- Head of a `leBytes` field. -/
theorem take4_leBytes (v : Nat) (r : List Nat) :
    (leBytes v 4 ++ r).take 4 = leBytes v 4 := by
  rw [List.take_append_of_le_length (by simp [leBytes_length]),
    List.take_of_length_le (by simp [leBytes_length])]

/-- Tail after a `leBytes` field. -/
theorem drop4_leBytes (v : Nat) (r : List Nat) :
    (leBytes v 4 ++ r).drop 4 = r := by
  rw [show (4 : Nat) = (leBytes v 4).length by simp [leBytes_length]]
  exact List.drop_left

/-- A slice inside the first of two appended lists. -/
theorem seg_slice_append {l r : List Nat} {o k : Nat}
    (h : o + k ≤ l.length) :
    ((l ++ r).drop o).take k = (l.drop o).take k := by
  rw [List.drop_append_of_le_length (by omega),
    List.take_append_of_le_length]
  rw [List.length_drop]
  omega

/-- The `i`-th segment of a flatMap with constant segment
length. -/
theorem flatSeg_take {α : Type} (f : α → List Nat) (k : Nat) :
    ∀ (l : List α) (i : Nat) (hi : i < l.length),
    (∀ x ∈ l, (f x).length = k) →
    ((l.flatMap f).drop (k * i)).take k = f (l[i]'hi) := by
  intro l
  induction l with
  | nil => intro i hi _; simp at hi
  | cons a l ih =>
    intro i hi hlen
    cases i with
    | zero =>
      show (f a ++ l.flatMap f).take k = f a
      rw [List.take_append_of_le_length (by
          have hh := hlen a (.head _); omega),
        List.take_of_length_le (by
          have hh := hlen a (.head _); omega)]
    | succ i =>
      have hi' : i < l.length := by
        have h := hi
        simp only [List.length_cons] at h
        omega
      have step : k * (i + 1) = k + k * i := by
        rw [Nat.left_distrib, Nat.mul_one, Nat.add_comm (k * i) k]
      rw [step, ← List.drop_drop]
      have hda : ((a :: l).flatMap f).drop k = l.flatMap f := by
        show (f a ++ l.flatMap f).drop k = l.flatMap f
        rw [← hlen a (.head _)]
        exact List.drop_left
      rw [hda, List.getElem_cons_succ]
      exact ih i hi' (fun x hx => hlen x (.tail _ hx))

/-- Encoded code length matches `codeBytes`. -/
theorem codeBytes_flatMap : ∀ (code : List Instr),
    (code.flatMap Instr.enc).length = codeBytes code
  | [] => rfl
  | x :: xs => by
    calc ((x :: xs).flatMap Instr.enc).length
        = (x.enc ++ xs.flatMap Instr.enc).length := by
          rw [List.flatMap_cons]
      _ = x.enc.length + codeBytes xs := by
          rw [List.length_append, codeBytes_flatMap xs]
      _ = codeBytes (x :: xs) := rfl

/-- `CodeSeq` from one read of the whole encoding. -/
theorem CodeSeq_of_rdN : ∀ (code : List Instr) {m : Mem} {c : Nat},
    m.rdN c (codeBytes code) = some (code.flatMap Instr.enc) →
    CodeSeq m c code := by
  intro code
  induction code with
  | nil => intro m c _; trivial
  | cons x xs ih =>
    intro m c h
    have hh : m.rdN c (x.enc.length + codeBytes xs) =
        some ((x :: xs).flatMap Instr.enc) := h
    have h1 : m.rdN c x.enc.length = some x.enc := by
      have hok1 : 0 + x.enc.length ≤ x.enc.length + codeBytes xs := by
        omega
      have hs := rdN_slice hh hok1
      simp only [Nat.add_zero, List.drop_zero] at hs
      rw [hs, List.flatMap_cons, List.take_left]
    have h2 : m.rdN (c + x.enc.length) (codeBytes xs) =
        some (xs.flatMap Instr.enc) := by
      have hok2 : x.enc.length + codeBytes xs ≤
          x.enc.length + codeBytes xs := Nat.le_refl _
      have hs := rdN_slice hh hok2
      have e : (xs.flatMap Instr.enc).take (codeBytes xs) =
          xs.flatMap Instr.enc := by
        rw [← codeBytes_flatMap xs]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, List.flatMap_cons, List.drop_left, e]
    exact ⟨h1, ih h2⟩

/-- Encoded annotation length. -/
theorem annflat_length : ∀ (ann : List (Nat × Height)),
    (ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
      leBytes x.2.a 4)).length = 12 * ann.length
  | [] => rfl
  | a :: l => by
    rw [List.flatMap_cons]
    simp only [List.length_append, leBytes_length,
      List.length_cons, annflat_length l]
    omega

/-- `AnnLayB` from one read of the whole annotation array. -/
theorem AnnLayB_of_rdN {m : Mem} {p : Nat}
    {ann : List (Nat × Height)}
    (h : m.rdN p (12 * ann.length) =
      some (ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4)))
    (hb : ∀ x ∈ ann, AnnBound x) : AnnLayB m p ann := by
  intro i hi
  have hseglen : ∀ x ∈ ann,
      ((fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4) x).length = 12 := fun x _ => by
        simp only [List.length_append, leBytes_length]
  have hseg : m.rdN (p + 12 * i) 12 =
      some (leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4) := by
    have hs := rdN_slice h (o := 12 * i) (k := 12) (by
      have : i + 1 ≤ ann.length := hi
      have e : 12 * i + 12 = 12 * (i + 1) := by omega
      omega)
    rw [hs, flatSeg_take _ 12 ann i hi hseglen]
  have hbnd := hb ann[i] (List.getElem_mem hi)
  obtain ⟨hb1, hb2, hb3⟩ := hbnd
  have hW : ann[i].1 < W :=
    Nat.lt_trans hb1 (by unfold W; decide)
  have f0 : m.rdN (p + 12 * i) 4 = some (leBytes ann[i].1 4) := by
    have hs := rdN_slice hseg (o := 0) (k := 4) (by omega)
    simp only [Nat.add_zero, List.drop_zero] at hs
    rw [hs]
    exact congrArg some (take4_leBytes _ _)
  have e1 : (leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4).drop 4 =
      leBytes ann[i].2.d 4 ++ leBytes ann[i].2.a 4 := by
    rw [List.append_assoc]
    exact drop4_leBytes _ _
  have f1 : m.rdN (p + 12 * i + 4) 4 = some (leBytes ann[i].2.d 4) := by
    have hs := rdN_slice hseg (o := 4) (k := 4) (by omega)
    rw [hs, e1]
    exact congrArg some (take4_leBytes _ _)
  have f2 : m.rdN (p + 12 * i + 8) 4 = some (leBytes ann[i].2.a 4) := by
    have hs := rdN_slice hseg (o := 8) (k := 4) (by omega)
    have e : (leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4).drop 8 = leBytes ann[i].2.a 4 := by
      show ((leBytes ann[i].1 4 ++ leBytes ann[i].2.d 4 ++
        leBytes ann[i].2.a 4).drop 4).drop 4 = leBytes ann[i].2.a 4
      rw [e1]
      exact drop4_leBytes _ _
    rw [hs, e]
    exact congrArg some
      (List.take_of_length_le (by simp [leBytes_length]))
  refine ⟨?_, ?_, ?_, hW⟩
  · rw [f0]; simp [leVal_leBytes, Nat.mod_eq_of_lt hb1]
  · rw [f1]; simp [leVal_leBytes, Nat.mod_eq_of_lt hb2]
  · rw [f2]; simp [leVal_leBytes, Nat.mod_eq_of_lt hb3]

/-- Encoded word length. -/
theorem wordEnc_length (wd : Word) :
    wd.enc.length = 20 + codeBytes wd.code + 12 * wd.ann.length := by
  show (leBytes wd.ins 4 ++ leBytes wd.outs 4 ++
    leBytes wd.locals 4 ++ leBytes wd.code.length 4 ++
    leBytes wd.ann.length 4 ++ (wd.code.flatMap Instr.enc) ++
    (wd.ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
      leBytes x.2.a 4))).length =
    20 + codeBytes wd.code + 12 * wd.ann.length
  simp only [List.length_append, leBytes_length, codeBytes_flatMap,
    annflat_length]

/-- Encoded words length matches `wordsBytes`. -/
theorem wordsBytes_flatMap : ∀ (wds : List Word),
    (wds.flatMap Word.enc).length = wordsBytes wds
  | [] => rfl
  | wd :: wds => by
    show ((wd :: wds).flatMap Word.enc).length =
      (20 + codeBytes wd.code + 12 * wd.ann.length) + wordsBytes wds
    rw [List.flatMap_cons, List.length_append, wordEnc_length,
      wordsBytes_flatMap wds]

/-- A word encoding, right-associated. -/
theorem wordEnc_form (wd : Word) : wd.enc =
    leBytes wd.ins 4 ++ (leBytes wd.outs 4 ++ (leBytes wd.locals 4 ++
      (leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
        ((wd.code.flatMap Instr.enc) ++
          (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4))))))) := by
  unfold Word.enc
  simp only [List.append_assoc]

/-- The five header fields of a word encoding, as the parser
reads them from its first 20 bytes. -/
theorem word_fields (wd : Word) (hb : WordBound wd) :
    leVal ((wd.enc.take 20).take 4) % W = wd.ins ∧
    leVal (((wd.enc.take 20).drop 4).take 4) % W = wd.outs ∧
    leVal (((wd.enc.take 20).drop 8).take 4) % W = wd.locals ∧
    leVal (((wd.enc.take 20).drop 12).take 4) % W = wd.code.length ∧
    leVal (((wd.enc.take 20).drop 16).take 4) % W = wd.ann.length := by
  have hf := wordEnc_form wd
  have d4 : wd.enc.drop 4 =
      leBytes wd.outs 4 ++ (leBytes wd.locals 4 ++
        (leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
          ((wd.code.flatMap Instr.enc) ++
            (wd.ann.flatMap (fun x => leBytes x.1 4 ++
              leBytes x.2.d 4 ++ leBytes x.2.a 4)))))) := by
    rw [hf]; exact drop4_leBytes _ _
  have d8 : wd.enc.drop 8 =
      leBytes wd.locals 4 ++ (leBytes wd.code.length 4 ++
        (leBytes wd.ann.length 4 ++ ((wd.code.flatMap Instr.enc) ++
          (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4))))) := by
    have e : wd.enc.drop 8 = (wd.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  have d12 : wd.enc.drop 12 =
      leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
        ((wd.code.flatMap Instr.enc) ++
          (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4)))) := by
    have e : wd.enc.drop 12 = (wd.enc.drop 8).drop 4 :=
      (List.drop_drop (i := 4) (j := 8)).symm
    rw [e, d8]; exact drop4_leBytes _ _
  have d16 : wd.enc.drop 16 =
      leBytes wd.ann.length 4 ++ ((wd.code.flatMap Instr.enc) ++
        (wd.ann.flatMap (fun x => leBytes x.1 4 ++
          leBytes x.2.d 4 ++ leBytes x.2.a 4))) := by
    have e : wd.enc.drop 16 = (wd.enc.drop 12).drop 4 :=
      (List.drop_drop (i := 4) (j := 12)).symm
    rw [e, d12]; exact drop4_leBytes _ _
  have tk : ∀ (o : Nat), o ≤ 16 →
      ((wd.enc.take 20).drop o).take 4 = (wd.enc.drop o).take 4 := by
    intro o ho
    rw [List.drop_take, List.take_take, Nat.min_eq_left (by omega)]
  obtain ⟨hb1, hb2, hb3, hb4, hb5, -, -⟩ := hb
  refine ⟨?_, ?_, ?_, ?_, ?_⟩
  · rw [List.take_take, Nat.min_eq_left (by decide : 4 ≤ 20), hf,
      take4_leBytes]
    exact leVal4_W hb1
  · rw [tk 4 (by decide), d4, take4_leBytes]
    exact leVal4_W hb2
  · rw [tk 8 (by decide), d8, take4_leBytes]
    exact leVal4_W hb3
  · rw [tk 12 (by decide), d12, take4_leBytes]
    exact leVal4_W hb4
  · rw [tk 16 (by decide), d16, take4_leBytes]
    exact leVal4_W hb5

/-- Past the five header fields of a word encoding sit the code
and annotation encodings. -/
theorem wordEnc_drop20 (wd : Word) :
    wd.enc.drop 20 = (wd.code.flatMap Instr.enc) ++
      (wd.ann.flatMap (fun x => leBytes x.1 4 ++ leBytes x.2.d 4 ++
        leBytes x.2.a 4)) := by
  have hf := wordEnc_form wd
  have e : wd.enc.drop 20 = (wd.enc.drop 16).drop 4 :=
    (List.drop_drop (i := 4) (j := 16)).symm
  have d16 : wd.enc.drop 16 =
      leBytes wd.ann.length 4 ++ ((wd.code.flatMap Instr.enc) ++
        (wd.ann.flatMap (fun x => leBytes x.1 4 ++
          leBytes x.2.d 4 ++ leBytes x.2.a 4))) := by
    have e4 : wd.enc.drop 16 = (wd.enc.drop 12).drop 4 :=
      (List.drop_drop (i := 4) (j := 12)).symm
    have d12 : wd.enc.drop 12 =
        leBytes wd.code.length 4 ++ (leBytes wd.ann.length 4 ++
          ((wd.code.flatMap Instr.enc) ++
            (wd.ann.flatMap (fun x => leBytes x.1 4 ++
              leBytes x.2.d 4 ++ leBytes x.2.a 4)))) := by
      have e3 : wd.enc.drop 12 = (wd.enc.drop 8).drop 4 :=
        (List.drop_drop (i := 4) (j := 8)).symm
      have d8 : wd.enc.drop 8 =
          leBytes wd.locals 4 ++ (leBytes wd.code.length 4 ++
            (leBytes wd.ann.length 4 ++
              ((wd.code.flatMap Instr.enc) ++
                (wd.ann.flatMap (fun x => leBytes x.1 4 ++
                  leBytes x.2.d 4 ++ leBytes x.2.a 4))))) := by
        have e2 : wd.enc.drop 8 = (wd.enc.drop 4).drop 4 :=
          (List.drop_drop (i := 4) (j := 4)).symm
        have d4 : wd.enc.drop 4 =
            leBytes wd.outs 4 ++ (leBytes wd.locals 4 ++
              (leBytes wd.code.length 4 ++
                (leBytes wd.ann.length 4 ++
                  ((wd.code.flatMap Instr.enc) ++
                    (wd.ann.flatMap (fun x => leBytes x.1 4 ++
                      leBytes x.2.d 4 ++ leBytes x.2.a 4)))))) := by
          rw [hf]; exact drop4_leBytes _ _
        rw [e2, d4]; exact drop4_leBytes _ _
      rw [e3, d8]; exact drop4_leBytes _ _
    rw [e4, d12]; exact drop4_leBytes _ _
  rw [e, d16]
  exact drop4_leBytes _ _

/-- `WordsLay` from one read of the whole words encoding. -/
theorem WordsLay_of_rdN : ∀ (wds : List Word) {m : Mem} {c : Nat},
    m.rdN c (wordsBytes wds) = some (wds.flatMap Word.enc) →
    (∀ w ∈ wds, WordBound w) → WordsLay m c wds := by
  intro wds
  induction wds with
  | nil => intro m c _ _; trivial
  | cons wd wds ih =>
    intro m c h hb
    have hbwd := hb wd (.head _)
    have hh : m.rdN c
        ((20 + codeBytes wd.code + 12 * wd.ann.length) +
          wordsBytes wds) =
        some (wd.enc ++ wds.flatMap Word.enc) := by
      rwa [List.flatMap_cons] at h
    have h20 : m.rdN c 20 = some (wd.enc.take 20) := by
      have hok : 0 + 20 ≤ (20 + codeBytes wd.code +
          12 * wd.ann.length) + wordsBytes wds := by omega
      have hs := rdN_slice hh hok
      simp only [Nat.add_zero, List.drop_zero] at hs
      rw [hs, List.take_append_of_le_length (by
        rw [wordEnc_length]; omega)]
    have hcode : m.rdN (c + 20) (codeBytes wd.code) =
        some (wd.code.flatMap Instr.enc) := by
      have hok : 20 + codeBytes wd.code ≤ (20 + codeBytes wd.code +
          12 * wd.ann.length) + wordsBytes wds := by omega
      have hs := rdN_slice hh hok
      have e : (wd.code.flatMap Instr.enc).take (codeBytes wd.code) =
          wd.code.flatMap Instr.enc := by
        rw [← codeBytes_flatMap]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, seg_slice_append (l := wd.enc)
        (r := wds.flatMap Word.enc) (o := 20) (k := codeBytes wd.code)
        (by rw [wordEnc_length]; omega), wordEnc_drop20,
        List.take_append_of_le_length
          (by have hcc := codeBytes_flatMap wd.code; omega), e]
    have hann : m.rdN (c + 20 + codeBytes wd.code)
        (12 * wd.ann.length) =
        some (wd.ann.flatMap (fun x => leBytes x.1 4 ++
          leBytes x.2.d 4 ++ leBytes x.2.a 4)) := by
      have hok : (20 + codeBytes wd.code) + 12 * wd.ann.length ≤
          (20 + codeBytes wd.code + 12 * wd.ann.length) +
            wordsBytes wds := by omega
      have hs := rdN_slice hh hok
      have ec : c + (20 + codeBytes wd.code) =
          c + 20 + codeBytes wd.code := by omega
      rw [ec] at hs
      have edrop : (wd.enc ++ wds.flatMap Word.enc).drop
            (20 + codeBytes wd.code) =
          wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4) ++
            wds.flatMap Word.enc := by
        have e1 : (wd.enc ++ wds.flatMap Word.enc).drop
              (20 + codeBytes wd.code) =
            ((wd.enc ++ wds.flatMap Word.enc).drop 20).drop
              (codeBytes wd.code) := (List.drop_drop).symm
        have e2 : (wd.enc ++ wds.flatMap Word.enc).drop 20 =
            (wd.code.flatMap Instr.enc) ++
              ((wd.ann.flatMap (fun x => leBytes x.1 4 ++
                  leBytes x.2.d 4 ++ leBytes x.2.a 4)) ++
                wds.flatMap Word.enc) := by
          rw [List.drop_append_of_le_length (by
            rw [wordEnc_length]; omega), wordEnc_drop20,
            List.append_assoc]
        rw [e1, e2, ← codeBytes_flatMap]
        exact List.drop_left
      have e : (wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4)).take
          (12 * wd.ann.length) =
          wd.ann.flatMap (fun x => leBytes x.1 4 ++
            leBytes x.2.d 4 ++ leBytes x.2.a 4) := by
        rw [← annflat_length]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, edrop,
        List.take_append_of_le_length
          (by have haa := annflat_length wd.ann; omega), e]
    have hrest : m.rdN
        (c + 20 + codeBytes wd.code + 12 * wd.ann.length)
        (wordsBytes wds) = some (wds.flatMap Word.enc) := by
      have hok : (20 + codeBytes wd.code + 12 * wd.ann.length) +
            wordsBytes wds ≤ (20 + codeBytes wd.code +
              12 * wd.ann.length) + wordsBytes wds := Nat.le_refl _
      have hs := rdN_slice hh hok
      have ec : c + (20 + codeBytes wd.code + 12 * wd.ann.length) =
          c + 20 + codeBytes wd.code + 12 * wd.ann.length := by omega
      rw [ec] at hs
      have e : (wds.flatMap Word.enc).take (wordsBytes wds) =
          wds.flatMap Word.enc := by
        rw [← wordsBytes_flatMap]
        exact List.take_of_length_le (Nat.le_refl _)
      rw [hs, ← wordEnc_length wd, List.drop_left, e]
    obtain ⟨f1, f2, f3, f4, f5⟩ := word_fields wd hbwd
    refine ⟨⟨wd.enc.take 20, h20, f1, f2, f3, f4, f5⟩,
      CodeSeq_of_rdN _ hcode,
      AnnLayB_of_rdN hann (fun x hx => hbwd.2.2.2.2.2.2 x hx),
      ih hrest (fun w hw => hb w (.tail _ hw))⟩

/-- Buffer bytes consumed by a data segment list. -/
def dataBytes : List (List Nat) → Nat
  | [] => 0
  | d :: ds => (4 + d.length) + dataBytes ds

/-- Encoded data length matches `dataBytes`. -/
theorem dataBytes_flat : ∀ (ds : List (List Nat)),
    (ds.flatMap (fun d => leBytes d.length 4 ++ d)).length =
      dataBytes ds
  | [] => rfl
  | d :: ds => by
    show ((d :: ds).flatMap (fun d => leBytes d.length 4 ++
        d)).length = (4 + d.length) + dataBytes ds
    rw [List.flatMap_cons, List.length_append, List.length_append,
      leBytes_length, dataBytes_flat ds]

/-- `DataSeq` from one read of the whole data encoding. -/
theorem DataSeq_of_rdN : ∀ (ds : List (List Nat)) {m : Mem}
    {c : Nat},
    m.rdN c (dataBytes ds) =
      some (ds.flatMap (fun d => leBytes d.length 4 ++ d)) →
    (∀ d ∈ ds, d.length < 256 ^ 4) →
    DataSeq m c (ds.map List.length) (c + dataBytes ds) := by
  intro ds
  induction ds with
  | nil =>
    intro m c _ _
    show DataSeq m c [] (c + 0)
    rw [Nat.add_zero]
    exact DataSeq.nil c
  | cons d ds ih =>
    intro m c h hb
    have hdlen := hb d (.head _)
    have hh : m.rdN c ((4 + d.length) + dataBytes ds) =
        some ((leBytes d.length 4 ++ d) ++
          ds.flatMap (fun d => leBytes d.length 4 ++ d)) := by
      rwa [List.flatMap_cons] at h
    have hd : rd32 m c = some d.length := by
      have hok : 0 + 4 ≤ (4 + d.length) + dataBytes ds := by omega
      have hs := rdN_slice hh hok
      simp only [Nat.add_zero, List.drop_zero] at hs
      show (m.rdN c 4).map leVal = some d.length
      rw [hs, List.take_append_of_le_length (by
          simp only [List.length_append, leBytes_length]; omega),
        take4_leBytes]
      simp [leVal_leBytes, Nat.mod_eq_of_lt hdlen]
    have hrest : m.rdN (c + 4 + d.length) (dataBytes ds) =
        some (ds.flatMap (fun d => leBytes d.length 4 ++ d)) := by
      have hok : (4 + d.length) + dataBytes ds ≤
          (4 + d.length) + dataBytes ds := Nat.le_refl _
      have hs := rdN_slice hh hok
      have hdrop : ((leBytes d.length 4 ++ d) ++
            ds.flatMap (fun d => leBytes d.length 4 ++ d)).drop
          (4 + d.length) =
          ds.flatMap (fun d => leBytes d.length 4 ++ d) := by
        rw [show (4 + d.length) =
          (leBytes d.length 4 ++ d).length by
            simp only [List.length_append, leBytes_length]]
        exact List.drop_left
      have e : (ds.flatMap (fun d => leBytes d.length 4 ++
            d)).take (dataBytes ds) =
          ds.flatMap (fun d => leBytes d.length 4 ++ d) := by
        rw [← dataBytes_flat]
        exact List.take_of_length_le (Nat.le_refl _)
      have ec : c + (4 + d.length) = c + 4 + d.length := by omega
      rw [ec] at hs
      rw [hs, hdrop, e]
    show DataSeq m c (d.length :: ds.map List.length)
      (c + ((4 + d.length) + dataBytes ds))
    rw [show c + ((4 + d.length) + dataBytes ds) =
      (c + 4 + d.length) + dataBytes ds by omega]
    exact DataSeq.cons hd
      (ih hrest (fun x hx => hb x (.tail _ hx)))

/-- A program encoding, right-associated. -/
theorem enc_form (pr : Prog) : pr.enc =
    [0x53, 0x50, 0x52, 0x31] ++ (leBytes pr.main 4 ++
      (leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))))) := by
  unfold Prog.enc
  simp only [List.append_assoc]

/-- Encoded program length. -/
theorem enc_length (pr : Prog) :
    pr.enc.length = 16 + dataBytes pr.data + wordsBytes pr.words := by
  rw [enc_form]
  have hmagiclen : ([0x53, 0x50, 0x52, 0x31] : List Nat).length = 4 :=
    rfl
  simp only [List.length_append, leBytes_length, hmagiclen,
    dataBytes_flat, wordsBytes_flatMap]
  omega

/-- Past the sixteen header bytes of a program encoding sit the
data and words encodings. -/
theorem enc_drop16 (pr : Prog) :
    pr.enc.drop 16 =
      (pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
        (pr.words.flatMap Word.enc) := by
  have hf := enc_form pr
  have d4 : pr.enc.drop 4 =
      leBytes pr.main 4 ++ (leBytes pr.words.length 4 ++
        (leBytes pr.data.length 4 ++
          ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
            (pr.words.flatMap Word.enc)))) := by
    rw [hf, show (4 : Nat) =
      ([0x53, 0x50, 0x52, 0x31] : List Nat).length from rfl]
    exact List.drop_left
  have d8 : pr.enc.drop 8 =
      leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))) := by
    have e : pr.enc.drop 8 = (pr.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  have d12 : pr.enc.drop 12 =
      leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc)) := by
    have e : pr.enc.drop 12 = (pr.enc.drop 8).drop 4 :=
      (List.drop_drop (i := 4) (j := 8)).symm
    rw [e, d8]; exact drop4_leBytes _ _
  have e : pr.enc.drop 16 = (pr.enc.drop 12).drop 4 :=
    (List.drop_drop (i := 4) (j := 12)).symm
  rw [e, d12]
  exact drop4_leBytes _ _

/-- The header values of a program encoding, as the parser reads
them from its first 20 bytes, plus the drop-4/8/12 forms the
composition needs for the field loads. -/
theorem enc_header_vals (pr : Prog) :
    leVal ((pr.enc.take 20).take 4) % 2 ^ 32 = 827478099 ∧
    leVal (((pr.enc.take 20).drop 4).take 4) % 2 ^ 32 =
      pr.main % 2 ^ 32 ∧
    leVal (((pr.enc.take 20).drop 8).take 4) % 2 ^ 32 =
      pr.words.length % 2 ^ 32 ∧
    leVal (((pr.enc.take 20).drop 12).take 4) % 2 ^ 32 =
      pr.data.length % 2 ^ 32 := by
  have hf := enc_form pr
  have d4 : pr.enc.drop 4 =
      leBytes pr.main 4 ++ (leBytes pr.words.length 4 ++
        (leBytes pr.data.length 4 ++
          ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
            (pr.words.flatMap Word.enc)))) := by
    rw [hf, show (4 : Nat) =
      ([0x53, 0x50, 0x52, 0x31] : List Nat).length from rfl]
    exact List.drop_left
  have d8 : pr.enc.drop 8 =
      leBytes pr.words.length 4 ++ (leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc))) := by
    have e : pr.enc.drop 8 = (pr.enc.drop 4).drop 4 :=
      (List.drop_drop (i := 4) (j := 4)).symm
    rw [e, d4]; exact drop4_leBytes _ _
  have d12 : pr.enc.drop 12 =
      leBytes pr.data.length 4 ++
        ((pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) ++
          (pr.words.flatMap Word.enc)) := by
    have e : pr.enc.drop 12 = (pr.enc.drop 8).drop 4 :=
      (List.drop_drop (i := 4) (j := 8)).symm
    rw [e, d8]; exact drop4_leBytes _ _
  have tk : ∀ (o : Nat), o ≤ 16 →
      ((pr.enc.take 20).drop o).take 4 = (pr.enc.drop o).take 4 := by
    intro o ho
    rw [List.drop_take, List.take_take, Nat.min_eq_left (by omega)]
  refine ⟨?_, ?_, ?_, ?_⟩
  · rw [List.take_take, Nat.min_eq_left (by decide : 4 ≤ 20), hf,
      show (4 : Nat) = ([0x53, 0x50, 0x52, 0x31] : List Nat).length
        from rfl, List.take_left]
    decide
  · rw [tk 4 (by decide), d4, take4_leBytes]
    exact leVal4_32 _
  · rw [tk 8 (by decide), d8, take4_leBytes]
    exact leVal4_32 _
  · rw [tk 12 (by decide), d12, take4_leBytes]
    exact leVal4_32 _

/-- The first twenty bytes of a program in memory. -/
theorem enc_head20 {m : Mem} {B : Nat} {pr : Prog}
    (h : m.rdN B pr.enc.length = some pr.enc)
    (h20 : 20 ≤ pr.enc.length) :
    m.rdN B 20 = some (pr.enc.take 20) := by
  have hok : 0 + 20 ≤ pr.enc.length := h20
  have hs := rdN_slice h hok
  simpa using hs

/-- The data section of a program in memory. -/
theorem enc_data_slice {m : Mem} {B : Nat} {pr : Prog}
    (h : m.rdN B pr.enc.length = some pr.enc) :
    m.rdN (B + 16) (dataBytes pr.data) =
      some (pr.data.flatMap (fun d => leBytes d.length 4 ++ d)) := by
  have hok : 16 + dataBytes pr.data ≤ pr.enc.length := by
    rw [enc_length]; omega
  have hs := rdN_slice h hok
  have e : (pr.data.flatMap (fun d => leBytes d.length 4 ++
        d)).take (dataBytes pr.data) =
      pr.data.flatMap (fun d => leBytes d.length 4 ++ d) := by
    rw [← dataBytes_flat]
    exact List.take_of_length_le (Nat.le_refl _)
  rw [hs, enc_drop16,
    List.take_append_of_le_length (by
      have hcc := dataBytes_flat pr.data; omega), e]

/-- The words section of a program in memory. -/
theorem enc_words_slice {m : Mem} {B : Nat} {pr : Prog}
    (h : m.rdN B pr.enc.length = some pr.enc) :
    m.rdN (B + 16 + dataBytes pr.data) (wordsBytes pr.words) =
      some (pr.words.flatMap Word.enc) := by
  have hok : (16 + dataBytes pr.data) + wordsBytes pr.words ≤
      pr.enc.length := by
    rw [enc_length]; omega
  have hs := rdN_slice h hok
  have ec : B + (16 + dataBytes pr.data) =
      B + 16 + dataBytes pr.data := by omega
  rw [ec] at hs
  have edrop : pr.enc.drop (16 + dataBytes pr.data) =
      pr.words.flatMap Word.enc := by
    have e1 : pr.enc.drop (16 + dataBytes pr.data) =
        (pr.enc.drop 16).drop (dataBytes pr.data) :=
      (List.drop_drop (i := dataBytes pr.data) (j := 16)).symm
    rw [e1, enc_drop16, ← dataBytes_flat]
    exact List.drop_left
  have e : (pr.words.flatMap Word.enc).take (wordsBytes pr.words) =
      pr.words.flatMap Word.enc := by
    rw [← wordsBytes_flatMap]
    exact List.take_of_length_le (Nat.le_refl _)
  rw [hs, edrop, e]

/-- EXISTENCE TARGET: with room, some oracle makes the final
theorem non-vacuous.  The witness is `stage1Oracle f A` for the
`A` given by `Room`.  What remains is the loader-prefix invariant:
at each of the four `mmap`/`mremap` sites, the argument and region
shapes that make that placement legal, and that no other memory
syscall is reachable afterwards. -/
theorem stage1_oracle_exists_target
    (f path argv0 stdin : List Nat) (top : Nat)
    (hinit : Stage1InitOk f path argv0 top)
    (hsize : f.length ≤ 2 ^ 40) (hroom : Room f path argv0 top) :
    ∃ oracle, OracleAdmits (stage1Init f path argv0 stdin oracle top) ∧
      ¬ HitsNomem (stage1Init f path argv0 stdin oracle top) := by
  sorry

/-- A PROVISIONAL explicit step bound as a function of the file size.
The cubic term is a placeholder shape (walk instructions times
annotation lookups that scan the annotation array).  The constant is
not yet derived: it will be fixed last, from the per-region step
counts (loader prefix, parse, walk, handlers) as those lemmas land,
and this definition updated to the derived value. -/
def stage1Bound (f : List Nat) : Nat :=
  1000000 + 10000 * (f.length + 16) ^ 3

/-- What acceptance means semantically. -/
def Accepted (f : List Nat) : Prop :=
  ∃ p : Prog, Sporos.decode f = some p ∧ Sporos.checkProg p = true

/-- FINAL TARGET, no-memory-refusal case: for every file, path,
stdin, stack top and oracle meeting the hypotheses, the real stage-1
image reaches an exit within `stage1Bound f` steps; the exit is 0
exactly when `Sporos.decode` and `Sporos.checkProg` accept, and 1
exactly when they do not. -/
theorem stage1_final_target
    (f path argv0 stdin : List Nat) (oracle : List Choice) (top : Nat)
    (hinit : Stage1InitOk f path argv0 top)
    (horacle : OracleAdmits (stage1Init f path argv0 stdin oracle top))
    (hnomem : ¬ HitsNomem (stage1Init f path argv0 stdin oracle top)) :
    ((∃ k, k ≤ stage1Bound f ∧
        run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 0) ↔ Accepted f) ∧
    ((∃ k, k ≤ stage1Bound f ∧
        run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 1) ↔ ¬ Accepted f) ∧
    (∃ k, k ≤ stage1Bound f ∧
      (run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 0 ∨
       run stage1Im k
          (Out.ok (stage1Init f path argv0 stdin oracle top)) =
          Out.exit 1)) := by
  sorry

/-- FINAL TARGET, memory-refusal case: if the run reaches a memory
syscall whose next oracle choice is a refusal, then within the same
bound it exits 2.  (`Out.exit` carries only the status, so neither
target says anything about output bytes.)  Together with
`stage1_final_target` this says exit 2 happens only on oracle memory
refusal. -/
theorem stage1_final_exit2_target
    (f path argv0 stdin : List Nat) (oracle : List Choice) (top : Nat)
    (hinit : Stage1InitOk f path argv0 top)
    (horacle : OracleAdmits (stage1Init f path argv0 stdin oracle top))
    (hhit : HitsNomem (stage1Init f path argv0 stdin oracle top)) :
    ∃ k, k ≤ stage1Bound f ∧
      run stage1Im k
        (Out.ok (stage1Init f path argv0 stdin oracle top)) =
        Out.exit 2 := by
  sorry

end SporosX1
