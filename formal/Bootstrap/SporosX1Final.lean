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
head, and `Kern.count` drops it only when it is a count.  These two
lemmas are the generic half of `¬ HitsNomem` for a no-`nomem`
oracle; the remaining half is the loader-prefix invariant discussed
in the header. -/

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
