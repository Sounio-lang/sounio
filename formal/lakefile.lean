import Lake
open Lake DSL

package sounio where
  name := "sounio-formal"

lean_lib Sounio where
  roots := #[`ElfLinker, `TypeChecker, `Effects, `LinearTypes, `Epistemic,
             `EquivalenceTheory,
             `OctonionAlgebra, `FanoLabellingOrbits, `EpistemicGemm,
             `EffectLinear, `HessianAD, `SecondOrderGUM, `NonAssocHessian,
             `TypeCheckerSoundness, `KnowledgeArithmeticSoundness,
             `ChannelAssignmentSemantics, `GradientTopology,
             `GradientTopologyBridge,
             `Scheduler.CriticalPathLowerBound,
             `Scheduler.PortMatchingSound,
             `Scheduler.ClarkFoldDeterminism,
             `Scheduler.MicroarchInvarianceWeak,
             `Scheduler.IntegerFixedPointClark,
             `OntologyAlignmentRepair,
             `OntologyClaimStatus,
             `OntologyEvolution,
             `OntologyRepairEquivalence,
             `OntologyEvolutionRepair,
             `ClaimStatusInterval,
             `OntologyMinimalRepair,
             `OntologyELReasoner,
             `OntologyELClosureVerified,
             `OntologyRepairTies,
             `OntologyELPlus,
             `OntologyELPlusClosureVerified,
             `OntologyELPlusClosureComplete,
             `OntologyELPlusNormalization,
             `TrustSemiring,
             `EpistemicSafety,
             `EpistemicStat,
             `EpistemicJoint]

-- Minimal x86-64 model of the bootstrap seed loaders (bootstrap/roots).
-- Not a default target: CI builds it by name.
-- One module per line, in alphabetical order, so that two pull requests adding
-- different modules touch different lines and merge without conflict. Every
-- module under Sporos/ is built by the glob, so a new Sporos module needs no
-- edit here at all.
lean_lib Bootstrap where
  roots := #[`Bootstrap, `Sporos]
  globs := #[
    .one `Bootstrap.Formats,
    .one `Bootstrap.Loop,
    .one `Bootstrap.Refine,
    .one `Bootstrap.SeedData,
    .one `Bootstrap.Seeds,
    .one `Bootstrap.X86Env,
    .one `Bootstrap.X86Min,
    .submodules `Sporos]

-- The two seeds on every listing (about 20 minutes of kernel time, the two in parallel); built by
-- bootstrap/model/seed_proofs.sh and the nightly job, not on every pull request.
lean_lib BootstrapSeeds where
  roots := #[`Bootstrap.SeedA, `Bootstrap.SeedB]
