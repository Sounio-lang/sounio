/-
TARGET STATEMENTS — registered only in the `BootstrapSporosStage1` library, which no
CI job builds; `bootstrap/model/stage1_proofs.sh` builds and audits it.

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
import Bootstrap.SporosX1FinalRefuse

open X86Env X86Int X86Dec Sporos

namespace SporosX1

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
  obtain ⟨A, -, -, -, -, hcase⟩ := stage1_oracle_prefix hinit hroom
  refine ⟨stage1Oracle f A, ?_,
    stage1Oracle_not_hitsNomem f path argv0 stdin A top⟩
  rcases hcase with ⟨k, hk⟩ |
    ⟨k, s', hrun, hrip, -, hregs', -, sp0, hr4, hsp0le, hshp⟩
  · exact oracleAdmits_of_exit hk
  · refine oracleAdmits_of_milestone hrun ?_
    intro k' s hk'le hr
    have hr' : run stage1Im (k' - k) (Out.ok s') = Out.ok s := by
      have e : k' = k + (k' - k) := by omega
      rw [e, run_add, hrun] at hr
      exact hr
    have hlen : 16 ≤ s'.regs.length := by
      obtain ⟨v, hv⟩ := hregs' 15 (by decide) (by decide)
      have h2 : s'.regs[15]? = some (some v) := by
        cases h3 : s'.regs[15]? with
        | none =>
          have h4 : (s'.regs[15]?).getD none = some v := hv
          rw [h3] at h4
          simp at h4
        | some w =>
          have h4 : (s'.regs[15]?).getD none = some v := hv
          rw [h3] at h4
          simp only [Option.getD_some] at h4
          rw [h4]
      obtain ⟨h5, -⟩ := List.getElem?_eq_some_iff.1 h2
      omega
    have hb : shapeBnd s'.mem := hshp
    exact chk_legal (chk_reachable hsp0le
      (Or.inl (chkMain_entry hrip hr4 hb hlen)) (k' - k) s hr')

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
  obtain ⟨k, hk, hrun⟩ := stage1_exit2_compose hinit horacle hhit
  refine ⟨k, ?_, hrun⟩
  have h1cube : f.length + 16 ≤ (f.length + 16) ^ 3 := by
    calc f.length + 16 = (f.length + 16) * 1 := by
          rw [Nat.mul_one]
      _ ≤ (f.length + 16) *
            ((f.length + 16) * (f.length + 16)) :=
          Nat.mul_le_mul_left _
            (Nat.mul_pos (by omega) (by omega))
      _ = (f.length + 16) ^ 3 := by
          rw [Nat.pow_succ, Nat.pow_succ, Nat.pow_succ,
            Nat.pow_zero, Nat.one_mul, Nat.mul_assoc]
  unfold stage1Bound
  omega


end SporosX1
