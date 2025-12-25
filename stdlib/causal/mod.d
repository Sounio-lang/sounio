// stdlib/causal/mod.d
// Causal Inference Module
//
// The world's first causal inference library that integrates Pearl's
// do-calculus with Bayesian epistemic uncertainty tracking.
//
// Submodules:
//   - core.d:       DAG representation, do-calculus, interventions
//   - discovery.d:  PC, FCI, and structure learning algorithms
//   - uplift.d:     CATE learners (S/T/X-learner) with uncertainty
//   - refutation.d: Robustness tests and sensitivity analysis
//
// Run individual module tests:
//   dc run stdlib/causal/core.d
//   dc run stdlib/causal/discovery.d
//   dc run stdlib/causal/uplift.d
//   dc run stdlib/causal/refutation.d

fn main() -> i32 {
    print("Causal Inference Module - stdlib/causal\n")
    print("========================================\n\n")

    print("Submodules:\n")
    print("  - core.d:       DAG, do-calculus, interventions\n")
    print("  - discovery.d:  Structure learning (PC, FCI)\n")
    print("  - uplift.d:     CATE learners with uncertainty\n")
    print("  - refutation.d: Robustness and sensitivity tests\n\n")

    print("Run individual tests:\n")
    print("  dc run stdlib/causal/core.d\n")
    print("  dc run stdlib/causal/discovery.d\n")
    print("  dc run stdlib/causal/uplift.d\n")
    print("  dc run stdlib/causal/refutation.d\n\n")

    print("Module loaded successfully.\n")
    0
}
