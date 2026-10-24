/-
Axiom report for the headline theorems of the `formal/` package.

Not a library root: it is run by `scripts/ci/lean_axiom_report.sh` in the CI
`lean-proofs` job (`lake build <modules>` then `lake env lean AxiomReport.lean`),
so the job log carries the real `#print axioms` output. formal/AXIOM_INVENTORY.md
points here instead of restating a result it cannot regenerate.
-/
import OntologyELPlusClosureComplete
import ElfLinker
import TypeChecker
import TrustSemiring
import EpistemicSafety
import EpistemicStat
import EpistemicJoint

-- EL+ saturation: complete and sound w.r.t. the derivation calculus `Der`.
#print axioms Sounio.OntologyELPlus.subBPlusC_iff
#print axioms Sounio.OntologyELPlus.conflictBPlusC_iff

-- ELF linker table (formal/README.md).
#print axioms Sounio.ElfLinker.sections_non_overlapping
#print axioms Sounio.ElfLinker.sections_offset_monotone
#print axioms Sounio.ElfLinker.section_align_respected
#print axioms Sounio.ElfLinker.symbol_within_section
#print axioms Sounio.ElfLinker.symbol_unique_name
#print axioms Sounio.ElfLinker.reloc_target_valid_thm
#print axioms Sounio.ElfLinker.reloc_offset_within_section
#print axioms Sounio.ElfLinker.reloc_symbol_valid

-- Type checker table (formal/README.md).
#print axioms Sounio.TypeChecker.subtype_refl
#print axioms Sounio.TypeChecker.subtype_trans
#print axioms Sounio.TypeChecker.knowledge_covariant
#print axioms Sounio.TypeChecker.fn_contravariant_arg
#print axioms Sounio.TypeChecker.fn_covariant_ret
#print axioms Sounio.TypeChecker.knowledge_unwrap_sub
#print axioms Sounio.TypeChecker.check_implies_infer
#print axioms Sounio.TypeChecker.no_effect_leakage

-- Provenance trust semiring (ADR-016 decision 12): tropical^3, and the rejected lexicographic design.
#print axioms Trust.distrib
#print axioms Trust.absorb
#print axioms Trust.plus_idem
#print axioms Trust.times_wf
#print axioms Trust.lex_not_distrib

-- Epistemic safety of the minimal Knowledge/Contest calculus (ADR-016 decisions 2 and 4).
#print axioms Epi.safety
#print axioms Epi.meet_no_silent
#print axioms Epi.meet_refuses
#print axioms Epi.resolve_identifiable
#print axioms Epi.resolve_refuses_both

-- Statistical layer: E_n policy, exact on squares (ADR-016 decision 3).
#print axioms Epi.Stat.meet_sound
#print axioms Epi.Stat.meet_refuses
#print axioms Epi.Stat.consistent_comm
#print axioms Epi.Stat.consistent_mono

-- Joint layers and the statistical contest (ADR-016 decisions 2-4).
#print axioms Epi.Joint.meet_sound
#print axioms Epi.Joint.meet_ok_interval
#print axioms Epi.Joint.meet_refuses_certified
#print axioms Epi.Joint.meet_refuses_statistical
#print axioms Epi.Joint.resolve_identifiable
#print axioms Epi.Joint.resolve_refuses_both
