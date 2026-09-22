# Sounio Testing Strategy

> **Canonical epistemic API.** Examples below use `epistemic::knowledge`
> (`ep_measured`, `ep_add`, `ep_val`, `ep_std`, `ep_confidence`), anchored by
> `tests/stdlib/epistemic/test_knowledge_madaros_import_e2e.sio`. Confidence is
> an `i64` on the 0..1000 scale, and `ep_add` keeps the minimum of the inputs'
> confidence. The legacy `epistemic_std` / `add_epistemic` names and
> `.value` / `.uncertainty` fields are not the checked surface.

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
use epistemic::knowledge::{ep_measured, ep_add, ep_val, ep_std, ep_confidence}

fn test_ep_add_basic() -> bool with Mut, Div, Panic {
    // Setup: ep_measured(val, std_dev) stores variance = std_dev^2,
    // confidence = 900.
    let a = ep_measured(10.0, 0.5)
    let b = ep_measured(20.0, 0.3)

    // Execute
    let result = ep_add(&a, &b)

    // Verify: value adds, variances add, confidence is the minimum.
    let value_ok = abs(ep_val(&result) - 30.0) < 0.0001
    let std_ok = abs(ep_std(&result) - 0.583) < 0.001
    let confidence_ok = ep_confidence(&result) == 900

    return value_ok && std_ok && confidence_ok
}

fn test_ep_add_edge_cases() -> bool {
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
use epistemic::knowledge::{Epistemic, ep_add, ep_val, ep_variance}

fn prop_add_commutative(a: Epistemic, b: Epistemic) -> bool {
    // ep_add(a, b) ≈ ep_add(b, a)
    let ab = ep_add(&a, &b)
    let ba = ep_add(&b, &a)

    return abs(ep_val(&ab) - ep_val(&ba)) < 0.0001
}

fn prop_variance_grows(a: Epistemic, b: Epistemic) -> bool {
    // Uncorrelated addition adds variances, so uncertainty never shrinks.
    let sum = ep_add(&a, &b)

    return ep_variance(&sum) >= ep_variance(&a) &&
           ep_variance(&sum) >= ep_variance(&b)
}
```

### 4. Performance Tests
**Purpose:** Ensure performance meets requirements

**Template:**
```sio
// tests/performance/epistemic_operations.sio
use epistemic::knowledge::{ep_measured, ep_add}

fn benchmark_ep_add() -> f64 {
    let iterations = 1000000
    let start = current_time()

    var i = 0
    while i < iterations {
        let a = ep_measured(10.0, 0.5)
        let b = ep_measured(20.0, 0.3)
        let _ = ep_add(&a, &b)
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

### Example: Testing `ep_add`

**Prompt:**
```
Generate comprehensive tests for ep_add.

Function signature: fn ep_add(a: &Epistemic, b: &Epistemic) -> Epistemic
(stdlib/epistemic/knowledge.sio; Epistemic is { val: f64, variance: f64, confidence: i64 })

Requirements:
1. Test normal cases: positive numbers, negative numbers, zero
2. Test edge cases: very large numbers, very small numbers, equal values
3. Test error conditions: NaN, infinity (if applicable)
4. Verify properties: commutativity, variance growth, confidence is the minimum

Expected behavior:
- ep_val(&result) = ep_val(&a) + ep_val(&b)
- ep_variance(&result) = ep_variance(&a) + ep_variance(&b)
- ep_confidence(&result) = min(ep_confidence(&a), ep_confidence(&b))

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
    TestCase { name: "ep_add_basic", function: test_ep_add_basic, category: "unit/epistemic" },
    TestCase { name: "ep_add_edge", function: test_ep_add_edge_cases, category: "unit/epistemic" },
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
use epistemic::knowledge::{ep_measured, ep_certain, ep_add, ep_val, ep_std, ep_confidence}

fn test_ep_add_table() -> bool with Mut, Div, Panic {
    // Rows: (a, b, expected value, expected std, expected confidence).
    // ep_measured stores confidence 900; ep_certain stores 1000.
    let a1 = ep_measured(1.0, 0.1)
    let b1 = ep_measured(2.0, 0.2)
    let a2 = ep_certain(0.0)
    let b2 = ep_measured(5.0, 0.5)

    let r1 = ep_add(&a1, &b1)
    if abs(ep_val(&r1) - 3.0) >= 0.0001 { return false }
    if abs(ep_std(&r1) - 0.224) >= 0.001 { return false }
    if ep_confidence(&r1) != 900 { return false }

    let r2 = ep_add(&a2, &b2)
    if abs(ep_val(&r2) - 5.0) >= 0.0001 { return false }
    if abs(ep_std(&r2) - 0.5) >= 0.001 { return false }
    if ep_confidence(&r2) != 900 { return false }

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
