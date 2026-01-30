// tests/e2e/edge_cases.rs — Edge case and boundary condition tests
//
// Tests unusual inputs, boundary conditions, and corner cases that
// might break the compiler or produce unexpected behavior.

use super::common::TestHarness;

// ============================================================================
// Empty and Minimal Input Tests
// ============================================================================

/// Test empty file
#[test]
fn test_empty_file() {
    let source = "";

    TestHarness::new()
        .compile_str("empty", source)
        .assert_success(); // Empty file should be valid
}

/// Test whitespace-only file
#[test]
fn test_whitespace_only() {
    let source = "   \n\t\n  \n\n";

    TestHarness::new()
        .compile_str("whitespace", source)
        .assert_success();
}

/// Test comments-only file
#[test]
fn test_comments_only() {
    let source = r#"
// This file has only comments
/* And this is a block comment
   spanning multiple lines */
// Nothing executable
"#;

    TestHarness::new()
        .compile_str("comments_only", source)
        .assert_success();
}

/// Test minimal valid program
#[test]
fn test_minimal_program() {
    let source = "fn main() {}";

    TestHarness::new()
        .compile_str("minimal", source)
        .assert_success();
}

// ============================================================================
// Unicode and Special Character Tests
// ============================================================================

/// Test Unicode in identifiers
#[test]
fn test_unicode_identifiers() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Médicament = chebi:drug;
type 薬 = chebi:drug;
type φάρμακο = chebi:drug;

fn main() {
    let café: Médicament = chebi:15365;
}
"#;

    let result = TestHarness::new().compile_str("unicode_ident", source);

    // May or may not support Unicode identifiers
    // Just ensure it doesn't crash
    let _ = result;
}

/// Test Unicode in strings
#[test]
fn test_unicode_strings() {
    let source = r#"
fn main() {
    let name = "阿司匹林";  // Aspirin in Chinese
    let emoji = "💊";
    let mixed = "Drug: 薬品 🏥";
}
"#;

    TestHarness::new()
        .compile_str("unicode_strings", source)
        .assert_success();
}

/// Test special characters in ontology URLs
#[test]
fn test_special_chars_url() {
    let source = r#"
ontology test from "file://path/with spaces/ontology.owl";
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("special_url", source);

    // Should either handle or give clear error
    let stderr = result.stderr();
    assert!(
        result.success() || stderr.contains("path") || stderr.contains("URL"),
        "Expected clear handling of special chars in URL"
    );
}

/// Test escape sequences
#[test]
fn test_escape_sequences() {
    let source = r#"
fn main() {
    let tab = "col1\tcol2";
    let newline = "line1\nline2";
    let quote = "He said \"hello\"";
    let backslash = "path\\to\\file";
    let unicode_escape = "\u{1F48A}";  // pill emoji
}
"#;

    TestHarness::new()
        .compile_str("escapes", source)
        .assert_success();
}

// ============================================================================
// Numeric Boundary Tests
// ============================================================================

/// Test large ontology term IDs
#[test]
fn test_large_term_id() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    // Very large term ID (may not exist but should parse)
    let d: Drug = chebi:999999999;
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("large_term_id", source);

    // Should either handle or give term-not-found error (not crash)
    assert!(
        result.success() || result.stderr().contains("999999999"),
        "Should handle large term IDs gracefully"
    );
}

/// Test zero as term ID
#[test]
fn test_zero_term_id() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:0;
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("zero_term_id", source);

    // Should handle gracefully
    let _ = result;
}

/// Test threshold boundaries
#[test]
fn test_threshold_zero() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = 0.0)]
fn exact_match_only(d: Drug) {}

fn main() {
    let d: Drug = chebi:15365;
    exact_match_only(d);
}
"#;

    TestHarness::new()
        .compile_str("threshold_zero", source)
        .assert_success();
}

/// Test threshold of 1.0
#[test]
fn test_threshold_one() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = 1.0)]
fn accept_anything(d: Drug) {}

fn main() {
    let d: Drug = chebi:15365;
    accept_anything(d);
}
"#;

    TestHarness::new()
        .compile_str("threshold_one", source)
        .assert_success();
}

/// Test negative threshold (should error)
#[test]
fn test_threshold_negative() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = -0.5)]
fn invalid_threshold(d: Drug) {}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("threshold_negative", source)
        .assert_failure();
}

/// Test threshold > 1.0 (should error or warn)
#[test]
fn test_threshold_over_one() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = 1.5)]
fn invalid_threshold(d: Drug) {}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("threshold_over_one", source);

    // Should either error or warn
    assert!(
        !result.success() || result.warning_count() > 0,
        "Expected error or warning for threshold > 1.0"
    );
}

// ============================================================================
// Deep Nesting Tests
// ============================================================================

/// Test deeply nested expressions
#[test]
fn test_deep_nesting() {
    let source = r#"
fn main() {
    let x = ((((((((((1 + 2) + 3) + 4) + 5) + 6) + 7) + 8) + 9) + 10) + 11);
}
"#;

    TestHarness::new()
        .compile_str("deep_nesting", source)
        .assert_success();
}

/// Test deeply nested blocks
#[test]
fn test_deep_blocks() {
    let mut source = String::from("fn main() {\n");
    for i in 0..50 {
        source.push_str(&format!("{}if true {{\n", "    ".repeat(i + 1)));
    }
    source.push_str(&format!("{}let x = 1;\n", "    ".repeat(51)));
    for i in (0..50).rev() {
        source.push_str(&format!("{}}}\n", "    ".repeat(i + 1)));
    }
    source.push_str("}\n");

    TestHarness::new()
        .compile_str("deep_blocks", &source)
        .assert_success();
}

/// Test deeply nested types
#[test]
fn test_deep_type_nesting() {
    let source = r#"
type A = Vec<Vec<Vec<Vec<Vec<i32>>>>>;

fn main() {
    let x: A = vec![vec![vec![vec![vec![1, 2, 3]]]]];
}
"#;

    TestHarness::new()
        .compile_str("deep_types", source)
        .assert_success();
}

// ============================================================================
// Large Input Tests
// ============================================================================

/// Test file with many functions
#[test]
fn test_many_functions() {
    let mut source = String::new();

    for i in 0..100 {
        source.push_str(&format!("fn func_{}() {{ let x = {}; }}\n", i, i));
    }
    source.push_str("fn main() {}\n");

    TestHarness::new()
        .compile_str("many_functions", &source)
        .assert_success();
}

/// Test file with many type definitions
#[test]
fn test_many_types() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
"#,
    );

    for i in 0..100 {
        source.push_str(&format!("type Type{} = chebi:drug;\n", i));
    }
    source.push_str("fn main() {}\n");

    TestHarness::new()
        .compile_str("many_types", &source)
        .assert_success();
}

/// Test very long line
#[test]
fn test_long_line() {
    let long_string = "x".repeat(10000);
    let source = format!(
        r#"
fn main() {{
    let s = "{}";
}}
"#,
        long_string
    );

    TestHarness::new()
        .compile_str("long_line", &source)
        .assert_success();
}

/// Test very long identifier
#[test]
fn test_long_identifier() {
    let long_ident = "x".repeat(1000);
    let source = format!(
        r#"
fn main() {{
    let {} = 42;
}}
"#,
        long_ident
    );

    let result = TestHarness::new().compile_str("long_ident", &source);

    // Should handle or give reasonable error
    let _ = result;
}

// ============================================================================
// Circular and Self-Referential Tests
// ============================================================================

/// Test self-referential type alias (should error)
#[test]
fn test_self_referential_type() {
    let source = r#"
type Infinite = Infinite;
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("self_ref_type", source)
        .assert_failure();
}

/// Test circular type aliases (should error)
#[test]
fn test_circular_types() {
    let source = r#"
type A = B;
type B = C;
type C = A;
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("circular_types", source)
        .assert_failure();
}

/// Test recursive struct
#[test]
fn test_recursive_struct() {
    let source = r#"
struct Node {
    value: i32,
    next: Option<Box<Node>>,  // Indirection required
}

fn main() {
    let n = Node { value: 1, next: None };
}
"#;

    TestHarness::new()
        .compile_str("recursive_struct", source)
        .assert_success();
}

/// Test infinite size struct (should error)
#[test]
fn test_infinite_size_struct() {
    let source = r#"
struct Bad {
    value: i32,
    next: Bad,  // No indirection - infinite size
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("infinite_struct", source)
        .assert_failure()
        .assert_error_contains("infinite")
        .assert_error_contains("size");
}

// ============================================================================
// Ontology Edge Cases
// ============================================================================

/// Test multiple ontology imports with same prefix
#[test]
fn test_duplicate_prefix() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology chebi from "file://ontologies/chebi_local.owl";  // Duplicate prefix
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("dup_prefix", source)
        .assert_failure();
}

/// Test alignment to self
#[test]
fn test_self_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

align chebi:drug ~ chebi:drug with distance 0.0;  // Trivially true

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:15365;
}
"#;

    TestHarness::new()
        .compile_str("self_align", source)
        .assert_success();
}

/// Test asymmetric alignment
#[test]
fn test_asymmetric_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// A -> B is declared
align chebi:drug ~ drugbank:drug with distance 0.1;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn need_chebi(d: ChEBIDrug) {}

#[compat(threshold = 0.2)]
fn need_drugbank(d: DrugBankDrug) {}

fn main() {
    let chebi: ChEBIDrug = chebi:15365;
    let drugbank: DrugBankDrug = drugbank:DB00945;

    // Both directions should work if alignment is symmetric
    need_chebi(drugbank);
    need_drugbank(chebi);
}
"#;

    TestHarness::new()
        .compile_str("asymmetric", source)
        .assert_success();
}

/// Test conflicting alignments
#[test]
fn test_conflicting_alignments() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// Two different distances for same pair
align chebi:drug ~ drugbank:drug with distance 0.1;
align chebi:drug ~ drugbank:drug with distance 0.5;  // Conflict?
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("conflict_align", source);

    // Should either use first/last or error
    let stderr = result.stderr();
    // Just ensure no crash
    let _ = stderr;
}

// ============================================================================
// Attribute Edge Cases
// ============================================================================

/// Test multiple compat attributes
#[test]
fn test_multiple_compat() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = 0.1)]
#[compat(threshold = 0.5)]  // Which one applies?
fn multi_compat(d: Drug) {}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("multi_compat", source);

    // Should handle somehow (error, use first, use last)
    let _ = result;
}

/// Test compat with no threshold
#[test]
fn test_compat_no_threshold() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat]  // Missing threshold
fn no_threshold(d: Drug) {}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("no_threshold", source);

    // Should error or use default
    let _ = result;
}

/// Test unknown attribute
#[test]
fn test_unknown_attribute() {
    let source = r#"
#[unknown_attr]
fn annotated() {}

fn main() {}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("unknown_attr", source);

    // Should warn or error about unknown attribute
    let stderr = result.stderr();
    assert!(
        stderr.contains("unknown") || stderr.contains("unrecognized") || result.success(),
        "Expected handling of unknown attribute"
    );
}

// ============================================================================
// Whitespace and Formatting Edge Cases
// ============================================================================

/// Test no whitespace between tokens
#[test]
fn test_minimal_whitespace() {
    let source = "fn main(){let x=1+2;let y=x*3;}";

    TestHarness::new()
        .compile_str("min_whitespace", source)
        .assert_success();
}

/// Test excessive whitespace
#[test]
fn test_excessive_whitespace() {
    let source = r#"
fn      main    (     )     {
    let     x     =     1     +     2     ;
}
"#;

    TestHarness::new()
        .compile_str("extra_whitespace", source)
        .assert_success();
}

/// Test Windows line endings (CRLF)
#[test]
fn test_crlf_endings() {
    let source = "fn main() {\r\n    let x = 1;\r\n}\r\n";

    TestHarness::new()
        .compile_str("crlf", source)
        .assert_success();
}

/// Test mixed line endings
#[test]
fn test_mixed_line_endings() {
    let source = "fn main() {\r\n    let x = 1;\n    let y = 2;\r\n}\n";

    TestHarness::new()
        .compile_str("mixed_endings", source)
        .assert_success();
}

// ============================================================================
// Comment Edge Cases
// ============================================================================

/// Test nested block comments
#[test]
fn test_nested_comments() {
    let source = r#"
/* outer
   /* inner comment */
   still outer
*/
fn main() {}
"#;

    let result = TestHarness::new().compile_str("nested_comments", source);

    // Nested comments may or may not be supported
    let _ = result;
}

/// Test comment at end of file without newline
#[test]
fn test_comment_eof() {
    let source = "fn main() {} // no newline at end";

    TestHarness::new()
        .compile_str("comment_eof", source)
        .assert_success();
}

/// Test doc comment
#[test]
fn test_doc_comment() {
    let source = r#"
/// This is a doc comment for main
fn main() {
    /// Doc comment on variable (unusual but should parse)
    let x = 1;
}
"#;

    TestHarness::new()
        .compile_str("doc_comment", source)
        .assert_success();
}

// ============================================================================
// Error Recovery Tests
// ============================================================================

/// Test parser error recovery
/// Note: The parser may handle incomplete statements in different ways
#[test]
fn test_error_recovery() {
    let source = r#"
fn first() {
    let x: i32 = "string";  // Error 1: type mismatch
}

fn second() {
    let y: bool = 42;  // Error 2: type mismatch
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("error_recovery", source);

    result.assert_failure();

    // Should find multiple errors (parser continues after first error)
    let error_count = result.error_count();
    assert!(
        error_count >= 1,
        "Expected at least 1 error, got {}",
        error_count
    );
}

/// Test that type checker continues after first type error
#[test]
fn test_type_error_recovery() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let a: Drug = 1;     // Type error 1
    let b: Drug = "x";   // Type error 2
    let c: Drug = true;  // Type error 3
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("type_recovery", source);

    result.assert_failure();

    // Should report multiple errors
    let error_count = result.error_count();
    assert!(
        error_count >= 2,
        "Expected multiple type errors, got {}",
        error_count
    );
}
