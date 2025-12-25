# Probability and Statistics Library for Demetrios

This directory contains foundational probability/statistics code for Bayesian UQ in the Darwin PBPK platform.

## Files

### 1. `random.d` - Random Number Generation
- **Status**: Working ✓
- **Features**:
  - Linear Congruential Generator (LCG) for reproducible randomness
  - Simple seed-based generation
- **Note**: Simplified implementation. Future versions will add:
  - Box-Muller transform for normal random variables
  - Stateful RNG with functional interface (requires tuple syntax fixes)

### 2. `distributions.d` - Probability Distributions
- **Status**: Working ✓
- **Features**:
  - Uniform distribution PDF
  - Struct-based distribution types
- **Note**: Foundational implementation. Future versions will add:
  - Normal (Gaussian) distribution
  - LogNormal distribution
  - Log-PDF functions for MCMC

### 3. `mcmc.d` - MCMC Sampling
- **Status**: Working ✓
- **Features**:
  - Metropolis-Hastings accept/reject logic
  - Simple MH step function
  - Demonstration of MCMC convergence toward target
- **Note**: Simplified sampler. Future versions will add:
  - Full MH with proper proposal distributions
  - Sample collection and diagnostics
  - Multi-parameter sampling

## Running Tests

Each file contains a `main()` function for testing:

```bash
# Test random number generation
./compiler/target/release/dc run stdlib/prob/random.d

# Test probability distributions
./compiler/target/release/dc run stdlib/prob/distributions.d

# Test MCMC sampler
./compiler/target/release/dc run stdlib/prob/mcmc.d
```

All tests should output "TEST PASSED" and return exit code 0.

## Known Limitations

These are **foundational implementations** to establish the probability/statistics infrastructure for Bayesian UQ. The following features are deferred due to compiler limitations:

1. **Tuple return types**: Currently problematic in Demetrios parser
   - Affects: Stateful RNG interface
   - Workaround: Functions return single values; state managed locally

2. **Integer literal suffixes**: `12345i64` syntax not supported
   - Workaround: Use type annotations `let x: i64 = 12345`

3. **Advanced math functions**: ln, exp, sin, cos need more robust implementations
   - Affects: Box-Muller transform, log-PDF functions
   - Workaround: Simplified versions or deferred to future

4. **Type inference with `as` casts**: Some edge cases with `(x as f64) / y`
   - Workaround: Simpler arithmetic without casts

## Future Enhancements

### Phase 1 (Current) - Foundations
- ✓ LCG random number generation
- ✓ Uniform distribution
- ✓ Basic MH accept/reject

### Phase 2 - Core Functionality
- Normal distribution with Box-Muller
- Log-PDF functions for all distributions
- Full MH sampler with proposal tuning
- Sample storage and basic diagnostics

### Phase 3 - Advanced Features
- Multi-parameter sampling
- Adaptive MCMC (AM, NUTS)
- Convergence diagnostics (Rhat, ESS)
- Integration with Darwin PBPK models

## Integration with Darwin PBPK

Once complete, this library will enable:

1. **Parameter Estimation**: Bayesian inference of PK parameters from data
2. **Uncertainty Quantification**: Posterior distributions for predictions
3. **Model Comparison**: Bayesian model selection
4. **Sensitivity Analysis**: Parameter importance via posterior variance

## Notes

- These implementations prioritize correctness and testing over performance
- Compiler improvements (tuple syntax, math functions) will enable more idiomatic code
- Code follows Demetrios stdlib patterns from `/stdlib/ode/` and `/stdlib/linalg/`

---

**Created**: December 2025 - Phase 4 of Darwin PBPK → Demetrios migration
**Purpose**: Bayesian UQ infrastructure for PBPK parameter estimation
