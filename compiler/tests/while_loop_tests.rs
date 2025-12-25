//! While Loop Regression Tests
//!
//! Tests for the while loop bug fix that ensures conditions are re-evaluated
//! on each iteration instead of being captured once at type-check time.
//!
//! Bug report: While loop conditions were desugared into a Loop with an
//! embedded If-Break, causing the condition to be evaluated only once.
//! This fix adds a proper While HIR node that re-evaluates the condition
//! fresh on each iteration.

use sounio::interp::{Interpreter, Value};

/// Helper to interpret source code and return the result
fn interpret(source: &str) -> Result<Value, String> {
    let tokens = sounio::lexer::lex(source).map_err(|e| format!("Lex error: {}", e))?;
    let ast =
        sounio::parser::parse(&tokens, source).map_err(|e| format!("Parse error: {}", e))?;
    let hir = sounio::check::check(&ast).map_err(|e| format!("Type error: {}", e))?;
    let mut interpreter = Interpreter::new();
    interpreter
        .interpret(&hir)
        .map_err(|e| format!("Runtime error: {}", e))
}

/// Helper to check the result is an integer
fn assert_result_int(source: &str, expected: i64) {
    match interpret(source) {
        Ok(Value::Int(n)) => assert_eq!(n, expected, "Expected {}, got {}", expected, n),
        Ok(v) => panic!("Expected Int({}), got {:?}", expected, v),
        Err(e) => panic!("Interpretation failed: {}", e),
    }
}

/// Helper to check the result is a float (within tolerance)
fn assert_result_float(source: &str, expected: f64, tolerance: f64) {
    match interpret(source) {
        Ok(Value::Float(f)) => {
            assert!(
                (f - expected).abs() < tolerance,
                "Expected {} (±{}), got {}",
                expected,
                tolerance,
                f
            );
        }
        Ok(v) => panic!("Expected Float({}), got {:?}", expected, v),
        Err(e) => panic!("Interpretation failed: {}", e),
    }
}

// ============================================================================
// Basic While Loop Tests
// ============================================================================

/// Test basic while loop counting up
/// This was the primary bug: the condition `i < 5` was not re-evaluated,
/// causing infinite loops or immediate exit.
#[test]
fn test_while_loop_count_up() {
    let source = r#"
fn main() -> i64 {
    let mut i = 0;
    while i < 5 {
        i = i + 1;
    }
    i
}
"#;
    assert_result_int(source, 5);
}

/// Test while loop counting down
#[test]
fn test_while_loop_count_down() {
    let source = r#"
fn main() -> i64 {
    let mut i = 10;
    while i > 0 {
        i = i - 1;
    }
    i
}
"#;
    assert_result_int(source, 0);
}

/// Test while loop with complex condition
#[test]
fn test_while_loop_complex_condition() {
    let source = r#"
fn main() -> i64 {
    let mut i = 0;
    let mut sum = 0;
    while i < 10 && sum < 30 {
        sum = sum + i;
        i = i + 1;
    }
    sum
}
"#;
    // sum: 0+1+2+3+4+5+6+7 = 28, then i=8, sum=28+8=36 > 30, so exits
    // Actually: 0+0=0, 1+1=1, 2+3=3, 3+6=6, 4+10=10, 5+15=15, 6+21=21, 7+28=28, 8+36=36 > 30
    assert_result_int(source, 36);
}

/// Test while loop that never enters (condition initially false)
#[test]
fn test_while_loop_never_enters() {
    let source = r#"
fn main() -> i64 {
    let mut x = 42;
    while false {
        x = 0;
    }
    x
}
"#;
    assert_result_int(source, 42);
}

/// Test while loop with single iteration
#[test]
fn test_while_loop_single_iteration() {
    let source = r#"
fn main() -> i64 {
    let mut done = false;
    let mut count = 0;
    while !done {
        count = count + 1;
        done = true;
    }
    count
}
"#;
    assert_result_int(source, 1);
}

// ============================================================================
// While Loop with Function Parameters (Bug #2)
// ============================================================================

/// Test while loop with function parameter in condition
/// This was bug #2: using parameters in while conditions caused hangs
/// because the parameter value was captured at definition time.
#[test]
fn test_while_with_parameter_in_condition() {
    let source = r#"
fn count_to(limit: i64) -> i64 {
    let mut i = 0;
    while i < limit {
        i = i + 1;
    }
    i
}

fn main() -> i64 {
    count_to(7)
}
"#;
    assert_result_int(source, 7);
}

/// Test while loop with multiple parameters
#[test]
fn test_while_with_multiple_parameters() {
    let source = r#"
fn count_range(start: i64, end: i64) -> i64 {
    let mut i = start;
    let mut count = 0;
    while i < end {
        count = count + 1;
        i = i + 1;
    }
    count
}

fn main() -> i64 {
    count_range(3, 10)
}
"#;
    assert_result_int(source, 7);
}

/// Test nested function calls with while loops
#[test]
fn test_while_nested_function_calls() {
    let source = r#"
fn sum_to(n: i64) -> i64 {
    let mut i = 1;
    let mut sum = 0;
    while i <= n {
        sum = sum + i;
        i = i + 1;
    }
    sum
}

fn main() -> i64 {
    sum_to(5) + sum_to(3)
}
"#;
    // sum_to(5) = 1+2+3+4+5 = 15
    // sum_to(3) = 1+2+3 = 6
    assert_result_int(source, 21);
}

// ============================================================================
// While Loop with Struct Field Updates (Bug #1)
// ============================================================================

/// Test while loop with struct field in condition
/// This was bug #1: struct field updates weren't reflected in conditions
/// due to struct values not being wrapped in Ref for mutation.
#[test]
fn test_while_with_struct_field_condition() {
    let source = r#"
struct Counter {
    value: i64,
    limit: i64
}

fn main() -> i64 {
    let mut c = Counter { value: 0, limit: 5 };
    while c.value < c.limit {
        c.value = c.value + 1;
    }
    c.value
}
"#;
    assert_result_int(source, 5);
}

/// Test while loop updating multiple struct fields
#[test]
fn test_while_with_multiple_struct_fields() {
    let source = r#"
struct State {
    x: i64,
    y: i64
}

fn main() -> i64 {
    let mut s = State { x: 0, y: 10 };
    while s.x < s.y {
        s.x = s.x + 1;
        s.y = s.y - 1;
    }
    s.x + s.y
}
"#;
    // x: 0,1,2,3,4,5
    // y: 10,9,8,7,6,5
    // exits when x >= y (x=5, y=5)
    assert_result_int(source, 10);
}

/// Test while loop with struct field and function parameter
#[test]
fn test_while_struct_field_with_parameter() {
    let source = r#"
struct Accumulator {
    total: i64
}

fn accumulate_to(limit: i64) -> i64 {
    let mut acc = Accumulator { total: 0 };
    let mut i = 1;
    while i <= limit {
        acc.total = acc.total + i;
        i = i + 1;
    }
    acc.total
}

fn main() -> i64 {
    accumulate_to(10)
}
"#;
    // 1+2+3+4+5+6+7+8+9+10 = 55
    assert_result_int(source, 55);
}

// ============================================================================
// While Loop with Break and Continue
// ============================================================================

/// Test while loop with break
#[test]
fn test_while_with_break() {
    let source = r#"
fn main() -> i64 {
    let mut i = 0;
    while true {
        i = i + 1;
        if i == 5 {
            break;
        }
    }
    i
}
"#;
    assert_result_int(source, 5);
}

/// Test while loop with continue
#[test]
fn test_while_with_continue() {
    let source = r#"
fn main() -> i64 {
    let mut i = 0;
    let mut sum = 0;
    while i < 10 {
        i = i + 1;
        if i % 2 == 0 {
            continue;
        }
        sum = sum + i;
    }
    sum
}
"#;
    // sum of odd numbers 1-9: 1+3+5+7+9 = 25
    assert_result_int(source, 25);
}

/// Test nested while loops with break
#[test]
fn test_nested_while_with_break() {
    let source = r#"
fn main() -> i64 {
    let mut outer = 0;
    let mut total = 0;
    while outer < 3 {
        let mut inner = 0;
        while inner < 5 {
            total = total + 1;
            inner = inner + 1;
            if inner == 3 {
                break;
            }
        }
        outer = outer + 1;
    }
    total
}
"#;
    // Each outer iteration does 3 inner iterations (breaks at 3)
    // 3 outer * 3 inner = 9
    assert_result_int(source, 9);
}

// ============================================================================
// While Loop Edge Cases
// ============================================================================

/// Test deeply nested while loops
#[test]
fn test_deeply_nested_while() {
    let source = r#"
fn main() -> i64 {
    let mut a = 0;
    let mut count = 0;
    while a < 3 {
        let mut b = 0;
        while b < 3 {
            let mut c = 0;
            while c < 3 {
                count = count + 1;
                c = c + 1;
            }
            b = b + 1;
        }
        a = a + 1;
    }
    count
}
"#;
    // 3 * 3 * 3 = 27
    assert_result_int(source, 27);
}

/// Test while loop with boolean variable condition
#[test]
fn test_while_boolean_variable_condition() {
    let source = r#"
fn main() -> i64 {
    let mut running = true;
    let mut count = 0;
    while running {
        count = count + 1;
        if count >= 10 {
            running = false;
        }
    }
    count
}
"#;
    assert_result_int(source, 10);
}

/// Test while loop in recursive function
#[test]
fn test_while_in_recursive_function() {
    let source = r#"
fn factorial(n: i64) -> i64 {
    let mut result = 1;
    let mut i = 2;
    while i <= n {
        result = result * i;
        i = i + 1;
    }
    result
}

fn main() -> i64 {
    factorial(5)
}
"#;
    // 5! = 120
    assert_result_int(source, 120);
}

/// Test while loop computing Fibonacci
#[test]
fn test_while_fibonacci() {
    let source = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 {
        return n;
    }
    let mut a = 0;
    let mut b = 1;
    let mut i = 2;
    while i <= n {
        let temp = a + b;
        a = b;
        b = temp;
        i = i + 1;
    }
    b
}

fn main() -> i64 {
    fib(10)
}
"#;
    // fib(10) = 55
    assert_result_int(source, 55);
}

/// Test while loop with early return
#[test]
fn test_while_with_early_return() {
    let source = r#"
fn find_first_multiple(n: i64, target: i64) -> i64 {
    let mut i = 1;
    while i <= 100 {
        if i * n >= target {
            return i;
        }
        i = i + 1;
    }
    0
}

fn main() -> i64 {
    find_first_multiple(7, 50)
}
"#;
    // First i where 7*i >= 50: i=8 (7*8=56)
    assert_result_int(source, 8);
}

// ============================================================================
// Regression Tests for Original Bug Reports
// ============================================================================

/// Regression test: Variable in condition must be re-evaluated
/// Original bug: `i < 5` was evaluated once, so loop ran forever or not at all
#[test]
fn test_regression_condition_reevaluation() {
    let source = r#"
fn main() -> i64 {
    let mut iterations = 0;
    let mut i = 0;
    while i < 100 {
        iterations = iterations + 1;
        i = i + 10;
    }
    iterations
}
"#;
    // i: 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
    // iterations: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
    assert_result_int(source, 10);
}

/// Regression test: Parameter shadowing must not break while loops
/// Original bug: Same name as function parameter caused scope issues
#[test]
fn test_regression_parameter_shadowing() {
    let source = r#"
fn process(count: i64) -> i64 {
    let mut count = count;  // Shadow the parameter
    while count > 0 {
        count = count - 1;
    }
    count
}

fn main() -> i64 {
    process(5)
}
"#;
    assert_result_int(source, 0);
}

/// Regression test: Struct field updates must be visible in condition
/// Original bug: Struct fields weren't mutable because structs weren't wrapped in Ref
#[test]
fn test_regression_struct_field_visibility() {
    let source = r#"
struct Timer {
    elapsed: i64,
    duration: i64
}

fn run_timer(duration: i64) -> i64 {
    let mut timer = Timer { elapsed: 0, duration: duration };
    while timer.elapsed < timer.duration {
        timer.elapsed = timer.elapsed + 1;
    }
    timer.elapsed
}

fn main() -> i64 {
    run_timer(10)
}
"#;
    assert_result_int(source, 10);
}
