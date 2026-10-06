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
