# Sounio Testing Strategy

## Philosophy

**Test-Driven Development via Specification:**
1. Write the test specification first
2. Generate implementation with AI
3. Verify against specification
4. Iterate until correct

## Test Categories

### 1. Unit Tests
**Purpose:** Verify individual functions work correctly

**Template:**
```sio
// tests/unit/epistemic/add_test.sio
fn test_add_epistemic_basic() -> bool {
    // Setup
    let a = epistemic_std(10.0, 0.5, 0.95)
    let b = epistemic_std(20.0, 0.3, 0.90)
    
    // Execute
    let result = add_epistemic(a, b)
    
    // Verify
    let value_ok = abs(result.value - 30.0) < 0.0001
    let uncertainty_ok = abs(result.uncertainty - 0.583) < 0.001
    let confidence_ok = result.confidence < 0.90  // Should decrease
    
    return value_ok && uncertainty_ok && confidence_ok
}

fn test_add_epistemic_edge_cases() -> bool {
    // Test zero, negative, large numbers, etc.
}
```

### 2. Integration Tests
**Purpose:** Verify components work together

**Template:**
```sio
// tests/integration/compiler_pipeline.sio
fn test_compile_hello_world() -> bool {
    // Write test program
    let source = "fn main() { println(\"Hello\") }"
    write_file("test_temp.sio", source)
    
    // Compile
    let exit_code = compile_file("test_temp.sio", "test_temp.out")
    
    // Execute
    let output = run_program("./test_temp.out")
    
    // Cleanup
    delete_file("test_temp.sio")
    delete_file("test_temp.out")
    
    return exit_code == 0 && output == "Hello\n"
}
```

### 3. Property-Based Tests
**Purpose:** Verify mathematical properties

**Template:**
```sio
// tests/property/epistemic_properties.sio
fn prop_add_commutative() -> bool {
    // For all epistemic values a, b:
    // add_epistemic(a, b) ≈ add_epistemic(b, a)
    
    let a = random_epistemic()
    let b = random_epistemic()
    
    let ab = add_epistemic(a, b)
    let ba = add_epistemic(b, a)
    
    return abs(ab.value - ba.value) < 0.0001
}

fn prop_uncertainty_grows() -> bool {
    // Adding increases uncertainty
    let a = random_epistemic()
    let b = random_epistemic()
    
    let sum = add_epistemic(a, b)
    
    return sum.uncertainty >= a.uncertainty && 
           sum.uncertainty >= b.uncertainty
}
```

### 4. Performance Tests
**Purpose:** Ensure performance meets requirements

**Template:**
```sio
// tests/performance/epistemic_operations.sio
fn benchmark_add_epistemic() -> f64 {
    let iterations = 1000000
    let start = current_time()
    
    var i = 0
    while i < iterations {
        let a = epistemic_std(10.0, 0.5, 0.95)
        let b = epistemic_std(20.0, 0.3, 0.90)
        let _ = add_epistemic(a, b)
        i = i + 1
    }
    
    let end = current_time()
    return (end - start) / iterations  // seconds per operation
}
```

### 5. Fuzz Tests
**Purpose:** Find edge cases and crashes

**Template:**
```sio
// tests/fuzz/parser_fuzz.sio
fn fuzz_parser() {
    let random_source = generate_random_source()
    
    // Should not crash
    let tokens = lex(random_source)
    let ast = parse(tokens)
    
    // Log if it doesn't crash
    log_fuzz_success(random_source)
}
```

## Test Organization

```
tests/
├── unit/                    # Individual functions
│   ├── epistemic/
│   ├── compiler/
│   └── stdlib/
├── integration/            # Component interactions
│   ├── compiler_pipeline/
│   ├── stdlib_modules/
│   └── end_to_end/
├── property/              # Mathematical properties
│   ├── epistemic_props/
│   └── numerical_props/
├── performance/           # Speed benchmarks
│   ├── compiler_speed/
│   └── runtime_speed/
├── fuzz/                  # Random input testing
│   ├── parser_fuzz/
│   └── typechecker_fuzz/
└── regression/            # Previously fixed bugs
    ├── bug_001/
    └── bug_002/
```

## Running Tests

### Manual Run:
```bash
# Run all tests
./run_tests.sh

# Run specific category
./run_tests.sh unit

# Run single test
./souc run tests/unit/epistemic/add_test.sio
```

### Continuous Integration:
```yaml
# .github/workflows/tests.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: ./build.sh
      - run: ./run_tests.sh
```

## Test Generation with AI

### Prompt Template for Test Generation:
```
Generate comprehensive tests for [MODULE/FUNCTION].

Function signature: [SIGNATURE]

Requirements:
1. Test normal cases: [EXAMPLES]
2. Test edge cases: [EDGE CASES]
3. Test error conditions: [ERRORS]
4. Verify properties: [PROPERTIES]

Expected behavior:
- [BEHAVIOR 1]
- [BEHAVIOR 2]

Generate test code in Sounio.
```

### Example: Testing `add_epistemic`

**Prompt:**
```
Generate comprehensive tests for add_epistemic function.

Function signature: fn add_epistemic(a: Epistemic<f64>, b: Epistemic<f64>) -> Epistemic<f64>

Requirements:
1. Test normal cases: positive numbers, negative numbers, zero
2. Test edge cases: very large numbers, very small numbers, equal values
3. Test error conditions: NaN, infinity (if applicable)
4. Verify properties: commutativity, uncertainty growth, confidence decrease

Expected behavior:
- Result value = a.value + b.value
- Result uncertainty = sqrt(a.uncertainty² + b.uncertainty²)
- Result confidence = min(a.confidence, b.confidence) * 0.99

Generate test code in Sounio.
```

## Test Maintenance

### 1. Test Registration
```sio
// tests/registry.sio
struct TestCase {
    name: string,
    function: fn() -> bool,
    category: string,
}

let all_tests: [TestCase] = [
    TestCase { name: "add_epistemic_basic", function: test_add_epistemic_basic, category: "unit/epistemic" },
    TestCase { name: "add_epistemic_edge", function: test_add_epistemic_edge_cases, category: "unit/epistemic" },
    // ...
]
```

### 2. Test Runner
```sio
// tests/runner.sio
fn run_all_tests() -> i32 {
    var passed = 0
    var failed = 0
    
    for test in all_tests {
        print("Running: " + test.name + "... ")
        
        let result = test.function()
        
        if result {
            println("PASS")
            passed = passed + 1
        } else {
            println("FAIL")
            failed = failed + 1
        }
    }
    
    println("\nSummary: " + int_to_string(passed) + " passed, " + int_to_string(failed) + " failed")
    
    if failed > 0 {
        return 1
    }
    return 0
}
```

### 3. Coverage Reporting
Track which code is tested:
```sio
// tests/coverage.sio
struct CoveragePoint {
    file: string,
    line: i64,
    function: string,
    tested: bool,
}

fn record_coverage(file: string, line: i64, function: string) {
    // Called from instrumented code
}
```

## Best Practices

1. **Write tests first** (Test-Driven Development)
2. **One assertion per test** (when possible)
3. **Test both success and failure**
4. **Use realistic data**
5. **Keep tests fast**
6. **Isolate tests** (no dependencies between tests)
7. **Document test purpose**
8. **Update tests when code changes**

## Common Test Patterns

### Pattern 1: Table-Driven Tests
```sio
fn test_add_epistemic_table() -> bool {
    let cases = [
        (epistemic_std(1.0, 0.1, 0.95), epistemic_std(2.0, 0.2, 0.90), 3.0, 0.224, 0.89),
        (epistemic_std(0.0, 0.0, 1.0), epistemic_std(5.0, 0.5, 0.95), 5.0, 0.5, 0.94),
        // ...
    ]
    
    for (a, b, expected_value, expected_uncertainty, expected_confidence) in cases {
        let result = add_epistemic(a, b)
        if !verify_result(result, expected_value, expected_uncertainty, expected_confidence) {
            return false
        }
    }
    return true
}
```

### Pattern 2: Golden Tests
```sio
fn test_compiler_golden() -> bool {
    let source = read_file("tests/golden/hello.sio")
    let expected = read_file("tests/golden/hello.expected")
    
    let output = compile_and_run(source)
    
    return output == expected
}
```

### Pattern 3: Mock Objects
```sio
struct MockFileSystem {
    files: Map<string, string>,
}

fn mock_read_file(fs: MockFileSystem, path: string) -> string {
    return fs.files.get(path) ?? ""
}
```

## Next Steps

1. **Start with unit tests** for core functions
2. **Add integration tests** for compiler pipeline
3. **Implement property tests** for mathematical correctness
4. **Set up CI** to run tests automatically
5. **Track coverage** to identify untested code

## Resources

- [Sounio Testing Examples](examples/)
- [Test Generation Prompts](docs/TEST_GENERATION_PROMPTS.md)
- [CI/CD Configuration](.github/workflows/)
