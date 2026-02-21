import Lake
open Lake DSL

package sounio where
  name := "sounio-formal"

lean_lib Sounio where
  roots := #[`ElfLinker, `TypeChecker, `Effects, `LinearTypes, `Epistemic,
             `OctonionAlgebra, `EpistemicGemm, `EffectLinear]
