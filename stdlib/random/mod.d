// random — Random Number Generation and Probability Distributions
//
// A comprehensive random number library for scientific computing with
// epistemic uncertainty awareness.
//
// Modules:
// - rng: Random number generators (PCG64, Xoshiro256++, SplitMix64)
// - distributions: Probability distributions (Normal, Gamma, Beta, etc.)
// - sampling: Random sampling, shuffling, and PK/PD variability
//
// Design Philosophy:
// - Functional approach: RNG state passed through and returned
// - Reproducible: Same seed → same sequence
// - High quality: PCG64 passes all statistical tests
// - Fast: Xoshiro256++ for simulation-heavy workloads
//
// Quick Start:
// ```d
// // Create RNG with seed
// var rng = pcg64_new(42)
//
// // Sample from Normal(0, 1)
// let n = normal_standard()
// let result = normal_sample(n, rng)
// rng = result.0
// let value = result.1
//
// // Sample from array
// let arr: [f64] = [1.0, 2.0, 3.0]
// let sample_result = sample_one_f64(arr, rng)
// ```
//
// PK/PD Example:
// ```d
// // Generate virtual population with IIV
// let iiv = iiv_typical()  // 30% CV
// let pop_result = generate_population(
//     10.0,   // pop_CL
//     50.0,   // pop_Vc
//     1.0,    // pop_ka
//     iiv,
//     1000,   // n_subjects
//     rng
// )
// let population = pop_result.1
// ```
//
// References:
// - O'Neill (2014): "PCG: A Family of Simple Fast Space-Efficient PRNGs"
// - Blackman & Vigna (2018): "Scrambled Linear Pseudorandom Number Generators"
// - Devroye (1986): "Non-Uniform Random Variate Generation"
// - Marsaglia & Tsang (2000): "A Simple Method for Generating Gamma Variables"
// - Lavielle (2014): "Mixed Effects Models for the Population Approach"

// ============================================================================
// RE-EXPORTS FROM SUBMODULES
// ============================================================================

// From rng.d:
// - SplitMix64, splitmix64_new, splitmix64_next
// - Pcg64, pcg64_new, pcg64_next_i64, pcg64_next_f64, pcg64_bounded
// - Xoshiro256, xoshiro256_new, xoshiro256_next_i64, xoshiro256_next_f64
// - RngState, rng_new, rng_next_f64, rng_next_i64, rng_next_bool, rng_bounded

// From distributions.d:
// - Uniform, uniform_new, uniform_sample, uniform_mean, uniform_variance
// - Normal, normal_new, normal_sample, normal_mean, normal_variance, normal_pdf
// - LogNormal, lognormal_new, lognormal_sample, lognormal_mean
// - Exponential, exponential_new, exponential_sample, exponential_mean
// - Gamma, gamma_new, gamma_sample, gamma_mean, gamma_variance
// - Beta, beta_new, beta_sample, beta_mean, beta_variance
// - Poisson, poisson_new, poisson_sample, poisson_mean
// - Bernoulli, bernoulli_new, bernoulli_sample

// From sampling.d:
// - sample_index, sample_one_f64, sample_one_i64
// - sample_weighted_index
// - shuffle_f64, shuffle_i64
// - resample_f64
// - IIV, iiv_new, iiv_typical
// - IndividualPK, generate_individual_pk, generate_population
// - add_proportional_error, add_additive_error, add_combined_error

fn main() -> i32 {
    print("random module\n")
    print("=============\n")
    print("\n")
    print("Submodules:\n")
    print("  rng.d          - Random number generators\n")
    print("  distributions.d - Probability distributions\n")
    print("  sampling.d     - Sampling and PK/PD variability\n")
    print("\n")
    print("Run individual modules for tests:\n")
    print("  dc run stdlib/random/rng.d\n")
    print("  dc run stdlib/random/distributions.d\n")
    print("  dc run stdlib/random/sampling.d\n")
    0
}
