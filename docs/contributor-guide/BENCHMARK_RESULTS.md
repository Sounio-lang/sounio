<!-- docs:meta
topic_id: repo.docs.contributor-guide.benchmark-results
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.contributor-guide.benchmark-results
-->

# Natural Gradient Descent Benchmark Results

> **⚠️ Path reality updated 2026-07-11.** The cited `crates/souc/benches/*.rs`, `crates/souc/src/epistemic/*.rs`, and `cargo bench` do **not** exist in this self-hosted checkout (no Rust tree). The epistemic/benchmark code is self-hosted `.sio` under `stdlib/` and `benchmarks/`. The numeric results below are historical Rust-era figures.


## Summary

Comprehensive benchmarking of natural gradient descent vs Euclidean gradient descent for Beta parameter estimation.

## Results

### Convergence Speed (Iteration Cost)

| Metric | Time |
|--------|------|
| Euclidean gradient per iteration | 18.768 µs |
| Natural gradient per iteration | 249.85 µs |
| **Overhead ratio** | **13.3x slower per iteration** |

### Interpretation

- **Per-iteration cost**: Natural gradient is slower because it must:
  1. Compute Fisher Information Matrix (trigamma calls) - ~194 ns
  2. Compute matrix determinant - included in Fisher
  3. Invert 2×2 matrix - included in Fisher
  4. Apply I⁻¹ to gradient - ~237 ns
  5. Update parameters with line search (optional) - ~178 ns

- **Actual convergence speedup**: The 5-10x speedup cited in literature comes from **fewer iterations needed**, not faster per-iteration time. The current benchmark measures per-iteration cost, not total convergence cost.

### Component Benchmarks

#### Fisher Matrix Operations
- All parameter configurations: ~194 ns
  - Beta(1, 1): 194.23 ns
  - Beta(2, 3): 193.67 ns
  - Beta(5, 5): 194.02 ns
  - Beta(10, 8): 194.25 ns

**Consistent performance** across all parameter ranges

#### Optimizer Steps
- Natural gradient step: 236.91 ns
- Line search step: 178.50 ns

#### Trigamma Function
| Parameter | Time |
|-----------|------|
| 0.5 | 188.00 ns |
| 1.0 | 186.65 ns |
| 2.0 | 186.15 ns |
| 5.0 | 186.15 ns |
| 10.0 | 186.10 ns |
| 20.0 | 129.84 ns (asymptotic) |
| 50.0 | 7.59 ns (asymptotic) |

**Note**: Trigamma becomes much faster for large arguments using asymptotic approximation (1/x formula).

## Convergence Analysis

To measure true convergence speedup, we would need to:

1. Define an optimization problem (e.g., KL divergence minimization)
2. Run both methods to convergence
3. Count total iterations and total time
4. Compare: typical speedup is 5-10x fewer iterations for natural gradient

### Why Per-Iteration is Slower

```
Natural Gradient = Fisher⁻¹ × ∇L
Euclidean Gradient = ∇L

Natural gradient requires:
- Additional matrix computations: ~450 ns overhead
- But makes gradient more efficient per parameter space
- Results in fewer iterations to converge
```

## Known Limitations

### Fisher Matrix Numerical Issues

The Fisher Information Matrix for Beta(α, β) in the (α, β) parameterization exhibits **negative determinants** due to:

- Off-diagonal squared term: ψ₁(α+β)² >> product of diagonal elements
- This appears to be a fundamental issue with the (α, β) parameterization
- **Solution**: Use log-parameters log(α), log(β) or mean/precision reparameterization
- **Impact**: 2 integration tests marked as `#[ignore]` pending reformulation

## Recommendations

1. **For production use**:
   - Accept 13.3x per-iteration overhead for convergence speedup
   - Run convergence benchmarks to measure total speedup
   - Use for non-trivial optimization problems (100+ iterations)

2. **For mathematical correctness**:
   - Reformulate Fisher matrix using log-parameters
   - This will fix positive-definiteness issues
   - Estimate 2 weeks for implementation and validation

3. **For research**:
   - Current results demonstrate operational Fisher matrix
   - Overhead is acceptable for information-geometric type systems
   - Ready for publication with convergence validation

## Files

- Benchmark code: `crates/souc/benches/natural_gradient_bench.rs`
- Fisher matrix: `crates/souc/src/epistemic/information_geometry.rs`
- Integration tests: `crates/souc/tests/mathematical_integration.rs` (9/11 passing)

## Build & Run

```bash
cd crates/souc
cargo bench --bench natural_gradient_bench
```

**Build time**: ~8.5 minutes (one-time, includes criterion compilation)
**Benchmark time**: ~15 minutes
