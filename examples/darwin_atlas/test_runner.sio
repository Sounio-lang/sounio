/// Test Runner for Darwin Atlas Kernels
///
/// Runs all module tests and reports results.

module test_runner

use operators::run_operator_tests;
use quaternion::run_quaternion_tests;
use exact_symmetry::run_symmetry_tests;
use approx_metric::run_metric_tests;

fn main() {
    print("=== Darwin Atlas Test Suite ===\n\n")

    var passed: usize = 0
    var failed: usize = 0

    // Test operators
    print("Testing operators... ")
    if run_operator_tests() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test quaternion
    print("Testing quaternion... ")
    if run_quaternion_tests() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test exact_symmetry
    print("Testing exact_symmetry... ")
    if run_symmetry_tests() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    // Test approx_metric
    print("Testing approx_metric... ")
    if run_metric_tests() {
        print("PASS\n")
        passed = passed + 1
    } else {
        print("FAIL\n")
        failed = failed + 1
    }

    print("\n=== Results ===\n")
    print("Passed: ")
    print_usize(passed)
    print("\nFailed: ")
    print_usize(failed)
    print("\n")

    if failed == 0 {
        print("\nAll tests passed!\n")
    } else {
        print("\nSome tests failed.\n")
    }
}

// Helper to print usize (since format! doesn't exist)
fn print_usize(n: usize) {
    if n == 0 {
        print("0")
        return
    }

    var digits: [u8] = []
    var val = n
    while val > 0 {
        digits.push((48 + (val % 10)) as u8)  // '0' = 48
        val = val / 10
    }

    // Print in reverse
    var i = digits.len()
    while i > 0 {
        i = i - 1
        print_char(digits[i])
    }
}

fn print_char(c: u8) {
    // Single character print
    let s: [u8; 1] = [c]
    print_bytes(&s)
}

// Placeholder - actual implementation depends on runtime
fn print_bytes(bytes: &[u8]) {
    // This would need runtime support
}
