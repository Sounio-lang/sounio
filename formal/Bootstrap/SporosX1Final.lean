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
