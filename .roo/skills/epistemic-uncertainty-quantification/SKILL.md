---
name: epistemic-uncertainty-quantification
description: Extend Sounio's epistemic types with advanced uncertainty quantification methods, such as confidence intervals, non‑Gaussian distributions, and Dempster‑Shafer evidence combination, positioning the language at the state‑of‑the‑art in measurement science.
---

# Epistemic Uncertainty Quantification

## When to use this skill

Use this skill when you need to:

- Add new uncertainty representations beyond variance‑based `Knowledge<T>` (e.g., confidence intervals, prediction intervals)
- Incorporate non‑Gaussian error models (e.g., skew‑normal, Student‑t, mixture distributions)
- Implement Dempster‑Shafer or other evidence‑combination frameworks
- Validate that the new quantification methods remain GUM‑compliant
- Extend the standard library with advanced epistemic operators

## When NOT to use this skill

- For basic uncertainty propagation that already works with `Knowledge<T>` (use the existing epistemic library)
- For changes unrelated to epistemic types (e.g., adding new numeric functions)
- For pure performance optimizations without semantic changes

## Inputs required

- Clear specification of the new uncertainty representation (mathematical definition)
- References to relevant literature (GUM supplements, statistical textbooks)
- Expected behavior for common operations (addition, multiplication, transformation)
- Where the new functionality should live (`stdlib/epistemic/` or a new submodule)

## Workflow

1. **Understand the existing epistemic foundation**
   - Read `stdlib/epistemic/knowledge.sio` and `stdlib/epistemic/gum.sio`
   - Review the GUM compliance proofs in `docs/compiler/EPISTEMIC_GUM_COMPLIANCE.md`
   - Examine the test suite `tests/epistemic/`

2. **Design the new representation**
   - Decide whether to extend `Knowledge<T>` or create a parallel type (e.g., `IntervalKnowledge<T>`)
   - Define the internal fields (point estimate, interval bounds, distribution parameters, etc.)
   - Specify how confidence updates and provenance merging work

3. **Implement the core type**
   - Create a new `.sio` file in `stdlib/epistemic/` (or a subdirectory)
   - Provide constructors, accessors, and basic arithmetic operations
   - Ensure uncertainty propagation follows the required mathematical rules

4. **Integrate with the type checker**
   - If the new type needs special checking rules, modify `self‑hosted/check/epistemic.sio`
   - Add any necessary compiler diagnostics for misuse

5. **Write tests**
   - Add run‑pass tests that verify the expected behavior
   - Add compile‑fail tests that enforce correct usage
   - Include GUM‑compliance verification for the new methods

6. **Update documentation**
   - Extend the epistemic library reference (`docs/reference/KNOWLEDGE_REFERENCE.md`)
   - Add examples showing the new functionality
   - Note any deviations from standard GUM (if applicable)

## Examples

### Adding confidence‑interval‑based knowledge

```sio
// New type: interval‑based epistemic value
pub struct IntervalKnowledge<T> {
    lower: T,
    upper: T,
    confidence: BetaConfidence,
    provenance: Provenance,
}

// Constructor from measurement with symmetric interval
pub fn measured_interval(
    center: T,
    half_width: T,
    confidence: BetaConfidence,
    instrument: string
) -> IntervalKnowledge<T> { ... }

// Propagation rules for addition (interval arithmetic)
pub fn add(a: IntervalKnowledge<f64>, b: IntervalKnowledge<f64>)
    -> IntervalKnowledge<f64> {
    IntervalKnowledge {
        lower: a.lower + b.lower,
        upper: a.upper + b.upper,
        confidence: a.confidence.combine(b.confidence).decay(0.9),
        provenance: a.provenance.merge(b.provenance),
    }
}
```

### Dempster‑Shafer combination

```sio
// Basic belief assignment
pub struct MassFunction {
    hypotheses: Vec<string>,
    masses: Vec<f64>,
}

// Combine two mass functions via Dempster's rule
pub fn combine_dempster(m1: MassFunction, m2: MassFunction)
    -> Result<MassFunction, ConflictError> { ... }
```

## Troubleshooting / edge cases

- **Numerical stability**: interval arithmetic can produce overly conservative bounds; consider affine arithmetic or probability‑bounds analysis.
- **Conflict in Dempster‑Shafer**: high conflict leads to counter‑intuitive results; provide alternative combination rules (Yager’s, Dubois‑Prade).
- **GUM compliance**: any new representation must either be a superset of GUM or clearly documented as an extension; ensure the existing GUM test suite still passes.

## Related files

- `stdlib/epistemic/knowledge.sio` – the core `Knowledge<T>` type
- `stdlib/epistemic/gum.sio` – GUM‑specific propagation formulas
- `self‑hosted/check/epistemic.sio` – epistemic type‑checking logic
- `docs/compiler/EPISTEMIC_GUM_COMPLIANCE.md` – compliance evidence

## Next steps

After implementing the new quantification method, consider:

- Adding visualisation helpers (e.g., plotting intervals with uncertainty bands)
- Integrating with refinement types to enforce domain‑specific constraints
- Creating a tutorial that contrasts the new method with classical variance‑based propagation