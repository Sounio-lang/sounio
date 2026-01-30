//! Tests for visibility enforcement during type checking (Phase 3b)

use sounio::check;
use sounio::lexer;
use sounio::parser;

// Test 1: Public function access should work
#[test]
fn test_public_function_access() {
    let source = r#"
        pub fn helper() -> i32 { 42 }
        fn main() -> i32 { helper() }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    // Should compile without visibility errors
    match result {
        Ok(_) => {} // Success - public function is accessible
        Err(e) => panic!("Expected public function to be accessible: {}", e),
    }
}

// Test 2: Public struct construction should work
#[test]
fn test_public_struct_construction() {
    let source = r#"
        pub struct Point {
            x: i32,
            y: i32,
        }

        fn main() -> Point {
            Point { x: 1, y: 2 }
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    // Should compile without visibility errors
    match result {
        Ok(_) => {} // Success - public struct is accessible
        Err(e) => panic!("Expected public struct to be accessible: {}", e),
    }
}

// Test 3: Built-in enum variants should always work
#[test]
fn test_builtin_enum_variants() {
    let source = r#"
        fn main() -> Option<i32> {
            let x: i32 = 42;
            Some(x)
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    // Should compile - built-in variants are always accessible
    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected built-in variants to be accessible: {}", e),
    }
}

// Test 4: Option constructor with type inference
#[test]
fn test_option_constructor() {
    let source = r#"
        fn get_option() -> Option<i32> {
            let val: i32 = 42;
            Some(val)
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    // Should compile - Some and Option are built-in
    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected Option to work: {}", e),
    }
}

// Test 5: Result constructor
#[test]
fn test_result_constructor() {
    let source = r#"
        fn get_result() -> Result<i32, bool> {
            let val: i32 = 42;
            Ok(val)
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    // Should compile - Ok is built-in
    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected Result to work: {}", e),
    }
}

// Test 6: Basic struct field assignment
#[test]
fn test_struct_field_initialization() {
    let source = r#"
        pub struct Data {
            value: i32,
        }

        fn create() -> Data {
            Data { value: 100 }
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected struct initialization to work: {}", e),
    }
}

// Test 7: Function call with arguments
#[test]
fn test_function_call_with_args() {
    let source = r#"
        pub fn add(a: i32, b: i32) -> i32 {
            a + b
        }

        fn main() -> i32 {
            add(1, 2)
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected public function with args to work: {}", e),
    }
}

// Test 8: Multiple public functions
#[test]
fn test_multiple_public_functions() {
    let source = r#"
        pub fn func1() -> i32 { 1 }
        pub fn func2() -> i32 { 2 }

        fn main() -> i32 {
            func1() + func2()
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected multiple public functions to work: {}", e),
    }
}

// Test 9: Nested struct in public struct
#[test]
fn test_nested_public_struct() {
    let source = r#"
        pub struct Inner {
            x: i32,
        }

        pub struct Outer {
            inner: Inner,
        }

        fn make_outer() -> Outer {
            Outer { inner: Inner { x: 42 } }
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected nested public structs to work: {}", e),
    }
}

// Test 10: Reference to public function through function pointer
#[test]
fn test_function_reference() {
    let source = r#"
        pub fn my_func() -> i32 { 42 }

        fn call_through_fn_type(f: fn() -> i32) -> i32 {
            f()
        }

        fn main() -> i32 {
            call_through_fn_type(my_func)
        }
    "#;

    let tokens = lexer::lex(source).expect("lex");
    let ast = parser::parse(&tokens, source).expect("parse");
    let result = check::check_ast(&ast);

    match result {
        Ok(_) => {} // Success
        Err(e) => panic!("Expected function reference to work: {}", e),
    }
}
