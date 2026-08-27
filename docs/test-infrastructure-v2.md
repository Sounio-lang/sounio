<!-- docs:meta
topic_id: repo.docs.test-infrastructure-v2
authority: repo_only
audience: users
last_validated: 2026-08-24
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.test-infrastructure-v2
-->

# Test Infrastructure V2

Enhanced test suite runner with parallel execution, JUnit XML output, and new test annotations.

## Quick Start

```bash
# Run all tests with parallel execution (default: use all CPUs)
bash scripts/dev/run_sio_test_suite_v2.sh

# Control parallelism
bash scripts/dev/run_sio_test_suite_v2.sh --jobs 8

# Generate JUnit XML for CI integration
bash scripts/dev/run_sio_test_suite_v2.sh --format junit

# Filter tests by name
bash scripts/dev/run_sio_test_suite_v2.sh --filter "kernel"

# Verbose output
bash scripts/dev/run_sio_test_suite_v2.sh --verbose
```

## Test Annotations

All annotations use the `//@` prefix in the test file header:

### Basic Annotations

| Annotation | Description |
|------------|-------------|
| `//@ run-pass` | Test should compile and execute successfully (exit 0) |
| `//@ compile-fail` | Test should fail to compile (non-zero exit) |
| `//@ ignore` | Skip this test entirely |
| `//@ check-only` | Only compile, don't execute (for run-pass tests) |
| `//@ expect-stdout: PATTERN` | Expected output pattern in stdout |
| `//@ expect-stdout-contains: PATTERN` | Substring that must appear in stdout (run-pass) |
| `//@ error-pattern: PATTERN` | Expected error message pattern |

### New Annotations (V2)

| Annotation | Description |
|------------|-------------|
| `//@ known-failure: REASON` | Documented accepted failure |
| `//@ skip-if: CONDITION` | Conditionally skip based on environment |
| `//@ requires: FEATURE` | Skip unless feature is available |
| `//@ flaky` | Mark as potentially unstable |
| `//@ timeout: SECONDS` | Override default timeout (default: 30s) |

### Skip-If Conditions

| Condition | Description |
|-----------|-------------|
| `no-gpu` | Skip unless GPU available |
| `no-llvm` | Skip unless LLVM available |
| `ci-only` | Skip when running in CI |

### Required Features

| Feature | Description |
|---------|-------------|
| `gpu` | Requires GPU support |
| `llvm` | Requires LLVM backend |

Unknown `expect-*` / `expected-*` header keys fail the test. The runner used
to skip them, so `//@ expect-stdout-contains:` asserted nothing: a run-pass
file that printed `FAIL` and exited 0 still went green. `skip-if` and
`requires` already fail closed on unrecognised values; stdout assertions now
match that contract.

## Test Result Categories

- **Pass**: Test passed
- **Fail**: Test failed unexpectedly
- **Known Failure (XFAIL)**: Failed but marked as known issue
- **Unexpected Pass (XPAS)**: Passed but marked as known failure (issue resolved!)
- **Skip**: Skipped due to annotations
- **Flaky**: Unstable test

## JUnit XML Output

Compatible with Jenkins, GitLab CI, GitHub Actions, Azure DevOps, CircleCI:

```bash
# Generate JUnit output
bash scripts/dev/run_sio_test_suite_v2.sh --format junit

# Output file location (default: ./test-results.xml)
export SOUNIO_TEST_JUNIT_FILE=/path/to/results.xml
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SOUNIO_TEST_JOBS` | Number of parallel jobs | `$(nproc)` |
| `SOUNIO_TEST_TIMEOUT` | Default timeout in seconds | `30` |
| `SOUNIO_TEST_JUNIT_FILE` | JUnit XML output path | `./test-results.xml` |
| `SOUNIO_GPU_AVAILABLE` | Mark GPU as available | unset |
| `SOUNIO_LLVM_AVAILABLE` | Mark LLVM as available | unset |
| `CI` | Mark CI environment | unset |

## Example Tests

### Known Failure

```sio
//@ compile-fail
//@ known-failure: Type checker allows invalid borrow (issue #123)

fn main() {
    let x = 5
    let y = &x
    // ... invalid operation
}
```

### GPU-Required Test

```sio
//@ run-pass
//@ requires: gpu
//@ timeout: 120

fn main() {
    // GPU kernel code
}
```

### CI-Only Skip

```sio
//@ run-pass
//@ skip-if: ci-only

fn main() {
    // Test that requires local environment
}
```

### Multiple Annotations

```sio
//@ run-pass
//@ timeout: 60
//@ flaky
//@ expect-stdout: success

fn main() {
    println("success")
}
```

## Performance

On a typical CI runner with 16 cores:

- **V1 (sequential)**: ~15 minutes for 709 tests
- **V2 (parallel)**: ~3 minutes for 709 tests (5x speedup)
