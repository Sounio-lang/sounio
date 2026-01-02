//! Property-based integration tests for the Sounio compiler
//!
//! This test file runs proptest-based tests to verify compiler invariants.

#![allow(dead_code)]

use proptest::prelude::*;
use sounio::{check, interp, lexer, parser};

// ============================================================================
// Reserved Keywords
// ============================================================================

/// Reserved keywords that cannot be used as identifiers
const RESERVED_KEYWORDS: &[&str] = &[
    // Basic keywords
    "as", "in", "if", "fn", "do", "or",
    // Control flow
    "for", "let", "var", "mut", "pub", "use", "mod", "ref",
    // Types and declarations
    "type", "enum", "impl", "self", "Self", "true", "false",
    "struct", "trait", "where", "const", "match", "while", "loop",
    "break", "return", "continue", "else", "move", "copy", "drop",
    // Effects and advanced features
    "with", "from", "async", "await", "spawn", "effect", "handler",
    "handle", "perform", "resume", "linear", "affine", "kernel",
    "tile", "device", "shared", "gpu", "sample", "observe", "infer",
    "proof", "ode", "pde", "causal", "nodes", "edges", "state",
    "params", "domain", "boundary", "initial", "dual", "grad",
    "import", "export", "module", "ontology", "align", "distance",
    "threshold", "compat", "extern", "equations", "query",
    // Built-in types
    "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128",
    "f32", "f64", "bool", "char", "str",
];

/// Check if a string is a reserved keyword
fn is_keyword(s: &str) -> bool {
    RESERVED_KEYWORDS.contains(&s)
}

// ============================================================================
// Arbitrary Generators (for future use with more complex tests)
// ============================================================================

/// Maximum depth for recursive structures to prevent stack overflow
const MAX_DEPTH: usize = 5;

/// Strategy for generating valid identifiers
fn arb_identifier() -> impl Strategy<Value = String> {
    prop::string::string_regex("[a-zA-Z_][a-zA-Z0-9_]{0,15}")
        .unwrap()
        .prop_map(|s| format!("id_{}", s))
}

/// Strategy for generating primitive types
fn arb_primitive_type() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("i32".to_string()),
        Just("i64".to_string()),
        Just("f32".to_string()),
        Just("f64".to_string()),
        Just("bool".to_string()),
        Just("string".to_string()),
        Just("()".to_string()),
    ]
}

/// Strategy for generating type expressions with bounded depth
fn arb_type_expr(depth: usize) -> BoxedStrategy<String> {
    if depth == 0 {
        arb_primitive_type().boxed()
    } else {
        prop_oneof![
            5 => arb_primitive_type(),
            1 => arb_type_expr(depth - 1).prop_map(|t| format!("&{}", t)),
            1 => arb_type_expr(depth - 1).prop_map(|t| format!("&!{}", t)),
            1 => (arb_type_expr(depth - 1), 1..16usize)
                .prop_map(|(t, n)| format!("[{}; {}]", t, n)),
        ]
        .boxed()
    }
}

/// Strategy for generating literal values
fn arb_literal() -> impl Strategy<Value = String> {
    prop_oneof![
        any::<i32>().prop_map(|n| n.to_string()),
        (0..1000i32, 0..100i32).prop_map(|(i, f)| format!("{}.{}", i, f)),
        prop::bool::ANY.prop_map(|b| if b { "true" } else { "false" }.to_string()),
    ]
}

/// Strategy for generating expressions with bounded depth
fn arb_expr(depth: usize) -> BoxedStrategy<String> {
    if depth == 0 {
        prop_oneof![arb_literal(), arb_identifier(),].boxed()
    } else {
        prop_oneof![
            5 => arb_literal(),
            3 => arb_identifier(),
            2 => (arb_expr(depth - 1), arb_expr(depth - 1))
                .prop_map(|(l, r)| format!("({} + {})", l, r)),
            1 => arb_expr(depth - 1).prop_map(|e| format!("({})", e)),
        ]
        .boxed()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if a source string can be lexed and parsed without errors
fn can_parse(source: &str) -> bool {
    match lexer::lex(source) {
        Ok(tokens) => parser::parse(&tokens, source).is_ok(),
        Err(_) => false,
    }
}

/// Check that the lexer doesn't panic
fn lexer_doesnt_panic(source: &str) -> bool {
    std::panic::catch_unwind(|| {
        let _ = lexer::lex(source);
    })
    .is_ok()
}

/// Check that the parser doesn't panic
fn parser_doesnt_panic(source: &str) -> bool {
    std::panic::catch_unwind(|| {
        if let Ok(tokens) = lexer::lex(source) {
            let _ = parser::parse(&tokens, source);
        }
    })
    .is_ok()
}

/// Check that the type checker doesn't panic
fn typecheck_doesnt_panic(source: &str) -> bool {
    std::panic::catch_unwind(|| {
        if let Ok(tokens) = lexer::lex(source) {
            if let Ok(ast) = parser::parse(&tokens, source) {
                let _ = check::check(&ast);
            }
        }
    })
    .is_ok()
}

/// Check if a source string successfully type-checks
fn typechecks_ok(source: &str) -> bool {
    match lexer::lex(source) {
        Ok(tokens) => match parser::parse(&tokens, source) {
            Ok(ast) => check::check(&ast).is_ok(),
            Err(_) => false,
        },
        Err(_) => false,
    }
}

/// Check that the interpreter doesn't panic
fn interpret_doesnt_panic(source: &str) -> bool {
    std::panic::catch_unwind(|| {
        if let Ok(tokens) = lexer::lex(source) {
            if let Ok(ast) = parser::parse(&tokens, source) {
                // check::check returns HIR
                if let Ok(hir) = check::check(&ast) {
                    let mut interpreter = interp::Interpreter::new();
                    let _ = interpreter.interpret(&hir);
                }
            }
        }
    })
    .is_ok()
}

// ============================================================================
// Parser Tests
// ============================================================================

/// Strategy for generating valid lowercase identifiers (not keywords)
fn valid_lowercase_ident() -> impl Strategy<Value = String> {
    "[a-z][a-z0-9_]{0,10}"
        .prop_filter("filter out keywords", |s| !is_keyword(s))
}

/// Strategy for generating valid uppercase identifiers (not keywords)
fn valid_uppercase_ident() -> impl Strategy<Value = String> {
    "[A-Z][a-zA-Z0-9_]{0,10}"
        .prop_filter("filter out keywords", |s| !is_keyword(s))
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn lexer_never_panics(source in ".*") {
        prop_assert!(lexer_doesnt_panic(&source));
    }

    #[test]
    fn parser_never_panics(source in ".*") {
        prop_assert!(parser_doesnt_panic(&source));
    }

    #[test]
    fn simple_let_parses(
        name in valid_lowercase_ident(),
        value in 0i32..1000,
    ) {
        let source = format!("let {} = {};", name, value);
        prop_assert!(can_parse(&source), "Failed to parse: {}", source);
    }

    #[test]
    fn simple_function_parses(
        name in valid_lowercase_ident(),
        ret_val in 0i32..1000,
    ) {
        // Sounio uses expression-based returns or `return` without semicolon
        let source = format!("fn {}() -> i32 {{ {} }}", name, ret_val);
        prop_assert!(can_parse(&source), "Failed to parse: {}", source);
    }

    #[test]
    fn struct_parses(
        name in valid_uppercase_ident(),
        field1 in valid_lowercase_ident(),
        field2 in valid_lowercase_ident(),
    ) {
        let source = format!("struct {} {{ {}: i32, {}: bool }}", name, field1, field2);
        prop_assert!(can_parse(&source), "Failed to parse: {}", source);
    }

    #[test]
    fn nested_parens_parse(depth in 1usize..20) {
        let open = "(".repeat(depth);
        let close = ")".repeat(depth);
        let source = format!("let x = {}42{};", open, close);
        prop_assert!(can_parse(&source), "Failed to parse nested parens: {}", source);
    }
}

// ============================================================================
// Type Checker Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn typecheck_never_panics_on_valid_syntax(
        name in valid_lowercase_ident(),
        value in 0i32..1000,
    ) {
        let source = format!("fn {}() {{ let x = {}; }}", name, value);
        prop_assert!(typecheck_doesnt_panic(&source));
    }

    #[test]
    fn arithmetic_typechecks(a in 0i64..100, b in 1i64..100) {
        // Sounio uses expression-based returns (last expression is the return value)
        // Integer literals default to i64 in Sounio
        let source = format!("fn test() -> i64 {{ {} + {} }}", a, b);
        prop_assert!(typecheck_doesnt_panic(&source));
        prop_assert!(typechecks_ok(&source), "Failed to typecheck: {}", source);
    }

    #[test]
    fn empty_functions_typecheck(name in valid_lowercase_ident()) {
        let source = format!("fn {}() {{ }}", name);
        prop_assert!(typecheck_doesnt_panic(&source));
        prop_assert!(typechecks_ok(&source), "Empty function should typecheck: {}", source);
    }

    #[test]
    fn typecheck_is_deterministic(
        name in valid_lowercase_ident(),
        value in 0i32..1000,
    ) {
        let source = format!("fn {}() {{ let x = {}; }}", name, value);

        let results: Vec<bool> = (0..5)
            .map(|_| typechecks_ok(&source))
            .collect();

        let first = results[0];
        for result in &results[1..] {
            prop_assert_eq!(*result, first, "Type checking not deterministic for: {}", source);
        }
    }
}

// ============================================================================
// Interpreter Tests
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn interpreter_never_panics(n in any::<i32>()) {
        // Sounio uses expression-based returns
        let source = format!("fn main() -> i32 {{ {} }}", n);
        prop_assert!(interpret_doesnt_panic(&source));
    }

    #[test]
    fn interpretation_is_deterministic(n in 0i32..1000) {
        // Sounio uses expression-based returns
        let source = format!("fn main() -> i32 {{ {} }}", n);

        // Interpret multiple times and check consistency
        let results: Vec<bool> = (0..3)
            .map(|_| interpret_doesnt_panic(&source))
            .collect();

        let first = results[0];
        for result in &results[1..] {
            prop_assert_eq!(*result, first, "Interpretation not consistent for: {}", source);
        }
    }
}

// ============================================================================
// Manual Tests
// ============================================================================

#[test]
fn test_basic_parsing() {
    assert!(can_parse("let x = 42;"));
    assert!(can_parse("fn foo() {}"));
    assert!(can_parse("struct Point { x: i32, y: i32 }"));
}

#[test]
fn test_empty_input() {
    assert!(lexer_doesnt_panic(""));
    assert!(parser_doesnt_panic(""));
}

#[test]
fn test_simple_typecheck() {
    // Sounio uses expression-based returns (last expression is the return value)
    // Integer literals default to i64 in Sounio
    assert!(typechecks_ok("fn test() -> i64 { 42 }"));
}

#[test]
fn test_type_mismatch_doesnt_panic() {
    // Type mismatch: returns i32 but declared as bool
    assert!(typecheck_doesnt_panic("fn test() -> bool { 42 }"));
}

#[test]
fn test_simple_interpretation() {
    // Sounio uses expression-based returns
    let source = "fn main() -> i32 { 42 }";
    assert!(interpret_doesnt_panic(source));
}
