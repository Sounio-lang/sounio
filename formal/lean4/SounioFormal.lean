-- formal/lean4/SounioFormal.lean
import SounioLinear
import SounioEffects
import SounioTyping
import SounioSemantics
import SounioRowPoly
import SounioUnits
import SounioProgress
import SounioSubstitution
import SounioPreservation
import SounioEpistemic
import SounioCausality

/-!
# Sounio Formal Verification — Phases 8–12

Lean 4 formalization of the Sounio linear-effect calculus.
No sorry. No Mathlib dependency. `lake build` completes in < 1 minute (25 jobs).

## Library inventory (12 modules)

| Library             | Lines | Theorems | Phase | Topic                                  |
|---------------------|-------|----------|-------|----------------------------------------|
| `SounioLinear`      |  459  |   ~40    |   8   | Four-modality lattice, structural rules|
| `SounioEffects`     |  637  |   ~50    |   8   | Algebraic effect rows, mask algebra    |
| `SounioTyping`      |  490  |   ~38    |  9–10 | Bidirectional typing judgment          |
| `SounioSemantics`   |  293  |    20    |  11   | CBV operational semantics              |
| `SounioRowPoly`     |  256  |    25    |  10   | Row-polymorphic effect variables       |
| `SounioUnits`       |  278  |    38    |  10   | Units of measure (dimensional analysis)|
| `SounioProgress`    |  557  |    51    |  12   | Canonical forms + progress theorem     |
| `SounioSubstitution`|  740  |    81    |  12   | Free variables, context ops, subst     |
| `SounioPreservation`|  678  |    72    |  12   | Type safety (Wright-Felleisen)         |
| `SounioEpistemic`   |  505  |    76    |  12   | Epistemic types (confidence lattice)   |
| `SounioCausality`   |  562  |    77    |  12   | Causal types (SCMs, do-calculus)       |
| **Total**           |**5539**|**~601** |       |                                        |

## Phase 12 highlights

### SounioProgress — Canonical Forms + Progress
- `canonical_fun_form` / `canonical_prod_form` / `canonical_bang_form`: value + type → shape
- `progress`: every closed well-typed term is a value or can step
- `progress_xor`: the dichotomy is exclusive (value ⊕ step, never both)
- `closed_normal_is_value`: normal forms of closed well-typed terms are values

### SounioSubstitution — Free Variables, Context Ops, Substitution
- `freeVars` / `freeVarsDec`: free variable computation + decidable membership
- `subst_not_free`: substituting a non-free variable is identity
- `typing_append_right`: context can be extended on the right
- `typing_closed_weakening`: closed terms can be typed in any context
- `substitution_closed_value`: substitution lemma for closed values (CBV)

### SounioPreservation — Type Safety (Wright-Felleisen Framework)
- `Stuck` / `Safe` predicates and their algebra
- `EvaluatesTo`: deterministic evaluation + uniqueness (`evaluatesTo_unique`)
- `MultiStepN`: counted reduction with determinism
- `Preserves` combinator: property preservation under reduction
- `type_safety`: parametric safety from progress + preservation
- Canonical forms re-derived independently from SounioProgress

### SounioEpistemic — Epistemic Type Theory (NOVEL)
- `Knowledge` type with confidence lattice and provenance tracking
- `confMeet` / `confJoin`: bounded lattice with full algebraic laws
- `combine` / `strengthen` / `derive` / `bayesianUpdate`: knowledge operations
- `knowledgeLeq`: partial order (refl, trans, antisymm)
- Uncertainty propagation, information content, degradation chains
- Connection to algebraic effect system (`epistemicRow`, `mask`, disjointness)

### SounioCausality — Causal Type Theory (NOVEL)
- `SCM`: structural causal models with `isTopological` (DAG) witness
- `Reachable`: ancestral relation with acyclicity + antisymmetry
- `intervene`: do-operator with 10+ properties (idempotent, commutative, topo-preserving)
- `dSepSimple`: d-separation criterion + monotonicity
- `counterfactualWorld`: multi-intervention worlds
- Pearl's do-calculus rules 1–3 (simplified)
- `backDoorCriterion` / `frontDoorCriterion` / `instrumentalVariable`
- Connection to effect system (`causalEffectRow`, `probCausalRow`)

## Out of scope (future work)

- Full subject reduction (requires linear context splitting in substitution)
- Type uniqueness / inversion (requires determinism of Sub+Weak)
- Normalization / strong normalization
- Denotational semantics
-/
