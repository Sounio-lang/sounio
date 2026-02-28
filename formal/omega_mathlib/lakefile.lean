import Lake
open Lake DSL

package «omega_mathlib» where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «OmegaMathlib» where
  roots := #[`OmegaMathlib.EpistemicPower, `OmegaMathlib.AccumulatorBounds, `OmegaMathlib.InternedChemicalGraph, `OmegaMathlib.EpistemicMapBound, `OmegaMathlib.GlycolysisEpistemic, `OmegaMathlib.EpistemicArena]
