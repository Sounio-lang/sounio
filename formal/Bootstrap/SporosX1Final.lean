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
    hb6mem, hb6k, hlenb6⟩

/-! ### Read-loop infrastructure -/

theorem step_mov32 {im : Nat → Option (Ins × Nat)} {s : X86Int.St}
    {d r len v : Nat} (hi : im s.rip = some (.mov32 d r, len))
    (hr : s.reg r = some v) :
    X86Int.step im s =
      .ok { s.set d (some (v % 2 ^ 32)) with rip := s.rip + len } := by
  simp only [X86Int.step, exec, hi, hr, lift, bind, Option.bind, pure]

theorem pages_mono {a b : Nat} (h : a ≤ b) :
    X86Env.pages a ≤ X86Env.pages b := by
  unfold X86Env.pages
  have hdiv : (a + page - 1) / page ≤ (b + page - 1) / page :=
    Nat.div_le_div_right (by omega)
  have hpage : 0 < page := by decide
  exact Nat.mul_le_mul_right page hdiv

theorem mem_wr_third {c1 c2 r : Region} {a v : Nat}
    (hdisj1 : c1.hi ≤ r.base) (hdisj2 : c2.hi ≤ r.base)
    (hge : r.base ≤ a) (hlt : a < r.hi) (hrw : r.rw = true) :
    Mem.wr [c1, c2, r] a v =
      some [c1, c2, { r with data := r.data.set (a - r.base) (v % 256) }] := by
  have hnot1 : ¬ (c1.base ≤ a ∧ a < c1.hi) := by omega
  have hnot2 : ¬ (c2.base ≤ a ∧ a < c2.hi) := by omega
  simp [Mem.wr, Mem.find, hnot1, hnot2, hge, hlt, hrw]

theorem wrN_third_some : ∀ (bs : List Nat) {c1 c2 r : Region} {a : Nat},
    c1.hi ≤ r.base → c2.hi ≤ r.base → r.rw = true → r.base ≤ a →
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
    obtain ⟨r'', hr'', hb, hh, hrw'⟩ :=
      wrN_third_some (c1 := c1) (c2 := c2)
        (r := { r with data := r.data.set (a - r.base) (b % 256) })
        (a := a + 1) t hdisj1 hdisj2 (by simpa using hrw)
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
theorem loader_read_once {f : List Nat} {A sp0 cap count : Nat}
    {c1 c2 buf : Region} {s : X86Int.St}
    (hrip : s.rip = base + 0xd7) (hlen16 : s.regs.length = 16)
    (hr3 : s.reg 3 = some A) (hr12 : s.reg 12 = some cap)
    (hr13 : s.reg 13 = some count) (hr14 : s.reg 14 = some 3)
    (hr4 : s.reg 4 = some (sp0 + 24))
    (hr0 : ∃ v, s.reg 0 = some v) (hr10 : ∃ v, s.reg 10 = some v)
    (hr8 : ∃ v, s.reg 8 = some v) (hr9 : ∃ v, s.reg 9 = some v)
    (hmem : s.mem = [c1, c2, buf]) (hbufbase : buf.base = A)
    (hbufrw : buf.rw = true) (hbufhi : buf.hi = A + cap)
    (hdisj1 : c1.hi ≤ A) (hdisj2 : c2.hi ≤ A)
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
  obtain ⟨buf', hwr, hbuf'b, hbuf'hi, hbuf'rw⟩ :=
    wrN_third_some (c1 := c1) (c2 := c2) (r := buf) (a := A + count)
      ((f.drop count).take n) (by omega) (by omega) hbufrw (by omega) (by
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
      ?_, ?_, ?_, ⟨buf', ?_, hbuf'b.trans hbufbase, hbuf'hi.trans hbufhi, hbuf'rw.trans hbufrw⟩, ?_, ?_⟩
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
      ?_, ?_, ?_, ?_, ?_, ?_, ?_, ⟨buf', ?_, hbuf'b.trans hbufbase, hbuf'hi.trans hbufhi, hbuf'rw.trans hbufrw⟩,
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
