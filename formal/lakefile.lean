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
             `EpistemicJoint, `EpistemicCorr]

-- Minimal x86-64 model of the bootstrap seed loaders (bootstrap/roots).
-- Not a default target: CI builds it by name.
-- Every module under Sporos/ is built by the glob, so a new Sporos module needs
-- no edit here at all. Bootstrap modules are listed one per line, in
-- alphabetical order: two pull requests that add modules in different
-- alphabetical gaps touch different lines and merge cleanly. Two that insert
-- into the same gap (say X86Float and X86Int, both between X86Env and X86Min)
-- can still conflict.
lean_lib Bootstrap where
  roots := #[`Bootstrap, `Sporos]
  globs := #[
    .one `Bootstrap.Formats,
    .one `Bootstrap.Loop,
    .one `Bootstrap.PDist,
    .one `Bootstrap.Refine,
    .one `Bootstrap.SeedData,
    .one `Bootstrap.Seeds,
    .one `Bootstrap.SporosX1,
    .one `Bootstrap.SporosX1Comp,
    .one `Bootstrap.SporosX1Image,
    .one `Bootstrap.TMR,
    .one `Bootstrap.X86Dec,
    .one `Bootstrap.TMRExposure,
    .one `Bootstrap.TMRVoter,
    .one `Bootstrap.X86Bridge,
    .one `Bootstrap.TMRX86,
    .one `Bootstrap.TMRWord,
    .one `Bootstrap.X86Env,
    .one `Bootstrap.X86Int,
    .one `Bootstrap.X86Min,
    .one `Bootstrap.X86Uni,
    .submodules `Sporos]

-- The two seeds on every listing (about 20 minutes of kernel time, the two in parallel), and
-- their lifting to the unified machine X86Uni (SeedUniBridge, which imports both); built by
-- bootstrap/model/seed_proofs.sh and the nightly job, not on every pull request.
lean_lib BootstrapSeeds where
  roots := #[`Bootstrap.SeedA, `Bootstrap.SeedB, `Bootstrap.SeedUniBridge]

-- The Sporos reference interpreter as a program (`sporos PROGRAM.spb < in > out`)
-- and the writer of the example programs (`sporos-examples DIR`).
lean_exe sporos where
  root := `Sporos.Main

lean_exe «sporos-examples» where
  root := `Sporos.Examples

lean_exe x86run where
  root := `Bootstrap.RunHex

lean_exe «x86int-diff» where
  root := `Bootstrap.X86IntDiffModel

lean_exe «x86env-diff» where
  root := `Bootstrap.X86EnvDiffModel

lean_exe «x86dec-fuzz» where
  root := `Bootstrap.X86DecFuzzModel

lean_exe «tmr-bound-check» where
  root := `Bootstrap.TMRBoundCheck

lean_exe «x86uni-diff» where
  root := `Bootstrap.X86UniDiffModel
