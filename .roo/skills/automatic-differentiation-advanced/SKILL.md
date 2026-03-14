---
name: automatic-differentiation-advanced
description: Extend Sounio's automatic differentiation capabilities to higher‑order derivatives, Hessians, differentiation through control flow, and GPU‑accelerated gradient computation, enabling state‑of‑the‑art scientific machine learning.
---

# Advanced Automatic Differentiation

## When to use this skill

Use this skill when you need to:

- Add support for higher‑order derivatives (second‑order, Hessians, Jacobians)
- Enable differentiation through loops, conditionals, and recursion
- Optimize gradient computation for GPU or distributed execution
- Integrate with existing neural‑network libraries (QNN, ONN, SNN)
- Implement novel AD techniques (checkpointing, source‑to‑source transformation)

## When NOT to use this skill

- For simple forward‑mode AD that already works (use the existing `autodiff` module)
- For changes unrelated to differentiation (e.g., adding new math functions)
- For performance tuning of non‑AD code

## Inputs required

- Specification of the AD extension (which order, which differentiation mode)
- Mathematical definitions of the new derivative operators
- Expected performance characteristics and memory usage
- Integration points with the existing AD infrastructure

## Workflow

1. **Understand the current AD implementation**
   - Read `stdlib/autodiff/` and `self‑hosted/check/autodiff.sio` (if present)
   - Review the AD test suite under `tests/autodiff/`
   - Examine the intermediate representation used for gradient computation

2. **Design the extension**
   - Decide whether to extend forward‑mode, reverse‑mode, or both
   - Determine how higher‑order derivatives will be represented (nested tangents, Hessian matrices)
   - Plan for control‑flow differentiation (e.g., taping loops, unrolling)
   - Consider memory/performance trade‑offs

3. **Implement the core differentiation logic**
   - Add new AD operators to the appropriate standard library modules
   - Extend the compiler's AD lowering passes if necessary
   - Ensure correctness for edge cases (discontinuities, non‑differentiable points)

4. **Integrate with GPU and parallel execution**
   - If targeting GPU, modify the GPU backend to support AD primitives
   - Add parallel gradient accumulation for distributed settings

5. **Write tests**
   - Add unit tests for each new differentiation operator
   - Include numerical correctness tests against finite‑difference approximations
   - Test differentiation through control‑flow constructs

6. **Update documentation**
   - Extend the AD user guide (`docs/qnn/AUTO_DIFFERENTIATION_GUIDE.md`)
   - Provide examples of higher‑order derivative usage
   - Note any limitations or performance considerations

## Examples

### Hessian computation via nested forward‑mode

```sio
// Function to differentiate
fn rosenbrock(x: f64, y: f64) -> f64 {
    (1.0 - x) * (1.0 - x) + 100.0 * (y - x * x) * (y - x * x)
}

// Compute Hessian using nested forward‑mode AD
let (value, grad, hess) = diff2_forward(rosenbrock)(0.5, 1.5)
// grad is (df/dx, df/dy)
// hess is [[d²f/dx², d²f/dxdy], [d²f/dydx, d²f/dy²]]
```

### Differentiation through a while loop

```sio
// Loop that accumulates a value
fn accumulate(n: i64) -> f64 {
    var i = 0
    var sum = 0.0
    while i < n {
        sum = sum + f(i)
        i = i + 1
    }
    return sum
}

// Gradient with respect to n (requires tape‑based reverse‑mode)
let dsum_dn = grad_reverse(accumulate)(10)
```

### GPU‑accelerated Jacobian

```sio
// Vector‑valued function
fn vector_func(x: [f64; 3]) -> [f64; 2] {
    [x[0] * x[1], sin(x[2])]
}

// Jacobian computed on GPU
kernel fn jacobian_kernel(x: [f64; 3]) -> [[f64; 3]; 2] with GPU {
    // GPU‑friendly Jacobian computation
}
```

## Troubleshooting / edge cases

- **Expression blow‑up**: higher‑order derivatives can produce huge intermediate expressions; consider simplification passes.
- **Control‑flow non‑differentiability**: loops with variable iteration counts may require special handling (e.g., unrolling up to a bound).
- **Numerical stability**: higher‑order finite‑difference validation may be noisy; use symbolic differentiation for verification where possible.
- **GPU memory**: reverse‑mode on GPU may need careful memory management for the tape.

## Related files

- `stdlib/autodiff/` – existing automatic differentiation modules
- `self‑hosted/check/autodiff.sio` – AD‑related type‑checking (if present)
- `tests/autodiff/` – AD test suite
- `docs/qnn/AUTO_DIFFERENTIATION_GUIDE.md` – AD user documentation

## Next steps

After implementing advanced AD, consider:

- Adding automatic batching (vmap) for vectorized gradient computation
- Integrating with optimisation libraries (gradient descent, Newton‑based solvers)
- Creating a differentiable programming tutorial that shows end‑to‑end scientific ML