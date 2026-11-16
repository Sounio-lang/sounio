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
lean_lib Bootstrap where
  roots := #[`Bootstrap.X86Min, `Bootstrap.SeedData, `Bootstrap.Seeds, `Bootstrap.Formats, `Bootstrap.Refine,
             `Bootstrap.Loop, `Bootstrap.X86Env,
             `Sporos.Core, `Sporos.Step, `Sporos.Check, `Sporos.Safe, `Sporos.SafeStep, `Sporos.Run,
             `Sporos.Dag, `Sporos.DagStep, `Sporos.DagSim, `Sporos.DagPrim, `Sporos.DagAlloc, `Sporos.DagRun,
             `Sporos.Witness]

-- The two seeds on every listing (about 20 minutes of kernel time, the two in parallel); built by
-- bootstrap/model/seed_proofs.sh and the nightly job, not on every pull request.
lean_lib BootstrapSeeds where
  roots := #[`Bootstrap.SeedA, `Bootstrap.SeedB]
