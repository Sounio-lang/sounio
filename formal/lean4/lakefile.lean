import Lake
open Lake DSL

package «SounioFormal» where

@[default_target]
lean_lib «SounioLinear» where

@[default_target]
lean_lib «SounioEffects» where

@[default_target]
lean_lib «SounioTyping» where

@[default_target]
lean_lib «SounioUnits» where

@[default_target]
lean_lib «SounioRowPoly» where

@[default_target]
lean_lib «SounioSemantics» where

@[default_target]
lean_lib «SounioEpistemic» where

@[default_target]
lean_lib «SounioProgress» where

@[default_target]
lean_lib «SounioSubstitution» where

@[default_target]
lean_lib «SounioPreservation» where

@[default_target]
lean_lib «SounioCausality» where

@[default_target]
lean_lib «SounioCayleyDickson» where

@[default_target]
lean_lib «SounioSkewCategory» where

@[default_target]
lean_lib «SounioBidirectionalBridge» where

@[default_target]
lean_lib «SounioFormal» where

-- The composition-independence obligation this PR introduces, discharged as
-- theorems rather than asserted: quadrature IS the rho = 0 case of the JCGM
-- combination law; it understates strictly under positive correlation; the
-- additive default this PR switches to is sound for every admissible rho and
-- tight at rho = +1; N fully-correlated steps combined in quadrature
-- understate by exactly sqrt(N); and the collider row that separates
-- d-separation from a reachability check.  Stated on VARIANCES so no square
-- root is constructed and core `omega` suffices -- Mathlib-free, same
-- discipline as SounioMeasConf.
@[default_target]
lean_lib «SounioIndepComposition» where
