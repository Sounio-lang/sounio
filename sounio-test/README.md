# sounio-test - Test Framework for Sounio

## 🎯 Overview

`sounio-test` is a comprehensive testing framework for the Sounio programming language, designed specifically for scientific and epistemic computing. It provides:

- **Epistemic testing** - Verify uncertainty propagation and provenance
- **Statistical testing** - Tests with statistical significance
- **Property-based testing** - Test properties rather than examples
- **Fuzzing for epistemic types** - Specialized fuzzing for Knowledge[T]
- **Uncertainty-aware assertions** - Assertions that consider confidence
- **CI/CD integration** - Works with existing Sounio validation system

## 🚀 Quick Start

### Installation
```bash
# As a dependency in your sounio.toml
[dependencies]
sounio-test = "^0.1"

# Or install globally for CLI usage
sounio-pkg add sounio-test --global
```

### Basic Usage
```sounio
// Import the test framework
use sounio_test::{test, assert, assert_epsilon, test_suite}

// Simple test
#[test]
fn test_addition() {
    let result = 2 + 2
    assert(result == 4, "2 + 2 should equal 4")
}

// Epistemic test
#[test]
fn test_uncertainty_propagation() {
    let x = Knowledge(value: 10.0, ε: 0.1, prov: "measurement")
    let y = Knowledge(value: 5.0, ε: 0.2, prov: "measurement")
    
    let result = x + y
    
    // Check value
    assert(result.value == 15.0, "Sum should be 15.0")
    
    // Check uncertainty propagation
    assert_epsilon(result.ε, 0.224, 0.001, 
        "Uncertainty should propagate correctly")
    
    // Check provenance
    assert(str_contains(result.prov, "addition"),
        "Provenance should mention addition")
}

// Statistical test
#[test(iterations = 1000, confidence = 0.95)]
fn test_random_distribution() {
    let sample = random_normal(mean = 0.0, std = 1.0)
    
    // Test that mean is within expected bounds
    assert_statistical(
        sample_mean(sample),
        expected = 0.0,
        tolerance = 0.1,
        confidence = 0.95
    )
}

// Run tests
fn main() with IO {
    run_tests()
}
```

### Running Tests
```bash
# Run all tests in current directory
sounio-test run

# Run specific test file
sounio-test run tests/my_tests.sio

# Run with verbose output
sounio-test run --verbose

# Run with minimum confidence requirement
sounio-test run --min-confidence 0.8

# Generate test report
sounio-test report --format html
```

## 📚 Test Types

### 1. Unit Tests
```sounio
#[test]
fn test_function() {
    // Test individual functions
}
```

### 2. Epistemic Tests
```sounio
#[test(uncertainty = true)]
fn test_with_uncertainty() {
    // Tests that verify uncertainty propagation
}
```

### 3. Statistical Tests
```sounio
#[test(iterations = 1000, confidence = 0.95)]
fn test_statistical_property() {
    // Tests that require statistical significance
}
```

### 4. Property-based Tests
```sounio
#[property]
fn addition_is_commutative(a: f64, b: f64) -> bool {
    a + b == b + a
}
```

### 5. Fuzz Tests
```sounio
#[fuzz]
fn test_division_safety(numerator: f64, denominator: f64) {
    if denominator != 0.0 {
        let result = numerator / denominator
        assert(!result.is_nan(), "Division should not produce NaN")
    }
}
```

### 6. Integration Tests
```sounio
#[integration_test]
fn test_pipeline() {
    // Test entire pipelines
}
```

## 🔧 Assertions

### Basic Assertions
```sounio
assert(condition, message)
assert_eq(actual, expected, message)
assert_ne(actual, expected, message)
assert_lt(actual, expected, message)
assert_gt(actual, expected, message)
```

### Epistemic Assertions
```sounio
// Assert with uncertainty tolerance
assert_epsilon(actual, expected, tolerance, message)

// Assert confidence level
assert_confidence(value, min_confidence, message)

// Assert provenance contains expected text
assert_provenance(value, expected_text, message)

// Assert uncertainty propagation
assert_uncertainty_propagated(result, operation, inputs)
```

### Statistical Assertions
```sounio
// Assert statistical property
assert_statistical(actual, expected, tolerance, confidence)

// Assert distribution property
assert_distribution(data, distribution, confidence)

// Assert independence
assert_independent(var1, var2, confidence)
```

### Floating Point Assertions
```sounio
// Assert with relative tolerance
assert_float_eq(actual, expected, rel_tol, abs_tol)

// Assert with ULPs (Units in Last Place)
assert_float_eq_ulps(actual, expected, max_ulps)
```

## 📊 Test Configuration

### Inline Configuration
```sounio
#[test(
    iterations = 1000,          // For statistical tests
    confidence = 0.95,         // Required confidence
    timeout = 5000,            // Timeout in milliseconds
    uncertainty = true,        // Enable uncertainty checking
    provenance = true,         // Check provenance
    tags = ["fast", "unit"]   // Test tags
)]
fn my_test() {
    // ...
}
```

### Configuration File (`sounio.test.toml`)
```toml
[default]
timeout = 10000
confidence_threshold = 0.8
check_provenance = true
check_uncertainty = true

[report]
format = "html"               # html, json, junit, markdown
output_dir = "./test-reports"
include_failed = true

[coverage]
enabled = true
min_coverage = 0.8
include_uncertainty = true

[fuzzing]
iterations = 10000
timeout_per_test = 1000
seed = 42  # For reproducibility

[statistical]
default_iterations = 1000
default_confidence = 0.95

[epistemic]
check_propagation = true
min_confidence_for_pass = 0.7
require_provenance = true

[[test_suite]]
name = "unit"
pattern = "tests/unit/**/*.sio"

[[test_suite]]
name = "integration"
pattern = "tests/integration/**/*.sio"

[[test_suite]]
name = "epistemic"
pattern = "tests/epistemic/**/*.sio"
requires = ["uncertainty", "provenance"]
```

## 🧪 Advanced Testing

### Property-based Testing
```sounio
use sounio_test::{property, for_all, integers, floats}

// Test commutative property
#[property]
fn addition_commutative(a: f64, b: f64) -> bool {
    a + b == b + a
}

// Test with generators
#[property]
fn multiplication_distributive(
    #[generator(integers(-100, 100))] a: i64,
    #[generator(integers(-100, 100))] b: i64,
    #[generator(integers(-100, 100))] c: i64
) -> bool {
    a * (b + c) == a * b + a * c
}

// Test epistemic properties
#[property]
fn uncertainty_adds(
    #[generator(epistemic_floats(0.0, 10.0, 0.1))] x: Knowledge[f64],
    #[generator(epistemic_floats(0.0, 10.0, 0.1))] y: Knowledge[f64]
) -> bool {
    let result = x + y
    // Uncertainty should increase or stay same
    result.ε >= max(x.ε, y.ε)
}
```

### Fuzzing
```sounio
use sounio_test::{fuzz, fuzzer}

// Basic fuzzing
#[fuzz]
fn test_division(numerator: f64, denominator: f64) {
    if denominator != 0.0 {
        let result = numerator / denominator
        assert(!result.is_nan() && !result.is_infinite(),
            "Division should produce valid number")
    }
}

// Epistemic fuzzing
#[fuzz]
fn test_epistemic_operations(
    #[fuzzer(epistemic_f64)] x: Knowledge[f64],
    #[fuzzer(epistemic_f64)] y: Knowledge[f64]
) {
    let sum = x + y
    let diff = x - y
    let prod = x * y
    
    // All results should have provenance
    assert(str_contains(sum.prov, "addition") || 
           str_contains(sum.prov, "arithmetic"))
    
    // Uncertainty should be reasonable
    assert(sum.ε >= 0.0 && sum.ε <= 1.0)
}
```

### Statistical Testing
```sounio
use sounio_test::{statistical_test, random_sample}

// Test random number generator
#[statistical_test(iterations = 10000, confidence = 0.99)]
fn test_random_uniform() {
    let samples = random_sample(size = 1000, 
        distribution = uniform(0.0, 1.0))
    
    // Test mean
    assert_statistical(
        mean(samples),
        expected = 0.5,
        tolerance = 0.05,
        confidence = 0.99
    )
    
    // Test variance
    assert_statistical(
        variance(samples),
        expected = 1.0 / 12.0,  // Variance of uniform(0,1)
        tolerance = 0.01,
        confidence = 0.99
    )
}
```

## 📈 Test Reports

### Console Output
```
🧪 Running Tests
================

✅ test_addition (0.2ms) ε=1.00
✅ test_uncertainty_propagation (1.5ms) ε=0.92
⚠️  test_statistical (15.2ms) ε=0.87 (below threshold 0.90)
❌ test_division_by_zero (0.1ms) ε=1.00

📊 Summary
==========
Total: 4 tests
Passed: 2 (ε≥0.90: 1, ε≥0.80: 1)
Warnings: 1 (low confidence)
Failed: 1
Confidence: 0.85

💡 Recommendations
=================
• Improve uncertainty propagation in test_statistical
• Fix division by zero handling
```

### HTML Report
Generates interactive HTML reports with:
- Test results with confidence levels
- Uncertainty visualization
- Provenance tracking
- Statistical significance charts
- Coverage reports

### JSON Report
```json
{
  "summary": {
    "total_tests": 42,
    "passed": 38,
    "failed": 2,
    "warnings": 2,
    "overall_confidence": 0.89
  },
  "tests": [
    {
      "name": "test_addition",
      "status": "passed",
      "confidence": 1.0,
      "duration_ms": 0.2,
      "provenance": "unit_test"
    }
  ]
}
```

## 🔗 Integration

### With sounio-pkg
```bash
# Run tests as part of build
sounio-pkg test

# Run with specific configuration
sounio-pkg test -- --min-confidence 0.9
```

### With CI/CD
```yaml
# GitHub Actions example
- name: Run tests
  run: sounio-test run --min-confidence 0.8

- name: Generate coverage report
  run: sounio-test coverage --html

- name: Upload test results
  uses: actions/upload-artifact@v3
  with:
    name: test-results
    path: test-reports/
```

### With LSP
Real-time test feedback in your editor:
- Test status indicators
- Quick fix suggestions
- Uncertainty visualization
- Provenance tracing

## 🏗️ Architecture

```
sounio-test/
├── src/
│   ├── runner.sio           # Test runner
│   ├── assertions.sio       # Assertion library
│   ├── generators.sio       # Test data generators
│   ├── statistical.sio      # Statistical testing
│   ├── fuzzer.sio          # Fuzzing engine
│   ├── coverage.sio        # Coverage tracking
│   └── report.sio          # Report generation
├── macros/                  # Compiler macros
├── plugins/                 # Editor plugins
└── examples/               # Example tests
```

## 🚧 Roadmap

### Phase 1 (MVP)
- [x] Basic test runner
- [x] Epistemic assertions
- [x] Statistical testing
- [x] Console reporting

### Phase 2
- [ ] Property-based testing
- [ ] Fuzzing engine
- [ ] HTML reports
- [ ] Coverage tracking

### Phase 3
- [ ] Visual uncertainty debugging
- [ ] Distributed testing
- [ ] Machine learning test generation
- [ ] Formal verification integration

### Phase 4
- [ ] Quantum test generation
- [ ] Causal testing
- [ ] Adversarial testing for AI
- [ ] Reproducibility certification

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE)
```