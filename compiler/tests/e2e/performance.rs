// tests/e2e/performance.rs — Performance and scalability tests
//
// Tests that the compiler:
// 1. Compiles in reasonable time
// 2. Handles large inputs
// 3. Doesn't have quadratic/exponential blowup

use super::common::TestHarness;
use std::time::Duration;

// ============================================================================
// Baseline Performance Tests
// ============================================================================

/// Test that a trivial program compiles quickly
#[test]
fn test_trivial_compile_time() {
    let source = "fn main() {}";

    TestHarness::new()
        .with_timing()
        .compile_str("trivial", source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(2));
}

/// Test simple program compile time
#[test]
fn test_simple_compile_time() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:15365;
}
"#;

    TestHarness::new()
        .with_timing()
        .compile_str("simple", source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(5));
}

/// Test moderate program compile time
#[test]
fn test_moderate_compile_time() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";
ontology mondo from "https://purl.obolibrary.org/obo/mondo.owl";

type Drug = chebi:drug;
type Phenotype = hp:phenotypic_abnormality;
type Disease = mondo:disease;

struct Patient {
    diseases: Vec<Disease>,
    phenotypes: Vec<Phenotype>,
    medications: Vec<Drug>,
}

fn analyze_patient(p: Patient) -> f64 {
    let mut score = 0.0;
    for d in p.diseases {
        score += 1.0;
    }
    for ph in p.phenotypes {
        score += 0.5;
    }
    score
}

fn check_interactions(drugs: Vec<Drug>) -> bool {
    drugs.len() < 5
}

fn main() {
    let patient = Patient {
        diseases: vec![mondo:0005015],
        phenotypes: vec![hp:0002315, hp:0001945],
        medications: vec![chebi:15365, chebi:5855],
    };

    let score = analyze_patient(patient);
    let safe = check_interactions(vec![chebi:15365]);
}
"#;

    TestHarness::new()
        .with_timing()
        .compile_str("moderate", source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(10));
}

// ============================================================================
// Scaling Tests - Functions
// ============================================================================

/// Test compile time scales linearly with function count
#[test]
fn test_function_scaling() {
    let counts = [10, 50, 100];
    let mut times = Vec::new();

    for count in counts {
        let mut source = String::new();
        for i in 0..count {
            source.push_str(&format!("fn func_{}(x: i32) -> i32 {{ x + {} }}\n", i, i));
        }
        source.push_str("fn main() {}\n");

        let result = TestHarness::new()
            .with_timing()
            .compile_str(&format!("funcs_{}", count), &source);

        result.assert_success();
        times.push((count, result.duration().unwrap()));
    }

    // Check that doubling input doesn't more than triple time (allow some overhead)
    // This catches O(n²) or worse complexity
    let (c1, t1) = times[0];
    let (c2, t2) = times[1];
    let (_c3, t3) = times[2];

    let ratio1 = t2.as_secs_f64() / t1.as_secs_f64();
    let _ratio2 = t3.as_secs_f64() / t2.as_secs_f64();
    let input_ratio = c2 as f64 / c1 as f64;

    // Allow up to 4x time for 5x input (sublinear is fine, quadratic is not)
    assert!(
        ratio1 < input_ratio * 2.0,
        "Function scaling appears superlinear: {} functions took {:?}, {} took {:?}",
        c1,
        t1,
        c2,
        t2
    );
}

/// Test with many small functions
#[test]
fn test_many_small_functions() {
    let mut source = String::new();
    for i in 0..500 {
        source.push_str(&format!("fn f{}() -> i32 {{ {} }}\n", i, i));
    }
    source.push_str("fn main() { f0(); f499(); }\n");

    TestHarness::new()
        .with_timing()
        .compile_str("many_small", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(30));
}

// ============================================================================
// Scaling Tests - Types
// ============================================================================

/// Test compile time scales with type count
#[test]
fn test_type_scaling() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
"#,
    );

    for i in 0..200 {
        source.push_str(&format!("type Type{} = chebi:drug;\n", i));
    }
    source.push_str("fn main() {}\n");

    TestHarness::new()
        .with_timing()
        .compile_str("many_types", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(15));
}

/// Test complex struct definitions
#[test]
fn test_complex_structs() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;
"#,
    );

    for i in 0..50 {
        source.push_str(&format!(
            r#"
struct Struct{} {{
    field_a: Drug,
    field_b: i32,
    field_c: f64,
    field_d: String,
    field_e: Vec<Drug>,
}}
"#,
            i
        ));
    }
    source.push_str("fn main() {}\n");

    TestHarness::new()
        .with_timing()
        .compile_str("complex_structs", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(15));
}

// ============================================================================
// Scaling Tests - Ontology Terms
// ============================================================================

/// Test compile time with many ontology term references
#[test]
fn test_term_reference_scaling() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let drugs: Vec<Drug> = vec![
"#,
    );

    for i in 0..200 {
        source.push_str(&format!("        chebi:{},\n", 15000 + i));
    }
    source.push_str("    ];\n}\n");

    TestHarness::new()
        .with_timing()
        .compile_str("many_terms", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(30));
}

/// Test with multiple ontologies
#[test]
fn test_multi_ontology_performance() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";
ontology mondo from "https://purl.obolibrary.org/obo/mondo.owl";
ontology uo from "https://purl.obolibrary.org/obo/uo.owl";

type Drug = chebi:drug;
type Process = go:biological_process;
type Phenotype = hp:phenotypic_abnormality;
type Disease = mondo:disease;
type Unit = uo:unit;

fn analyze(
    drug: Drug,
    process: Process,
    phenotype: Phenotype,
    disease: Disease,
    unit: Unit,
) -> bool {
    true
}

fn main() {
    analyze(
        chebi:15365,
        go:0006915,
        hp:0002315,
        mondo:0005015,
        uo:0000022,
    );
}
"#;

    TestHarness::new()
        .with_timing()
        .compile_str("multi_ontology", source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(20));
}

// ============================================================================
// Scaling Tests - Alignments
// ============================================================================

/// Test compile time with many alignments
#[test]
fn test_alignment_scaling() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";
"#,
    );

    // Add 100 alignments
    for i in 0..100 {
        source.push_str(&format!(
            "align chebi:{} ~ drugbank:DB{:05} with distance 0.1;\n",
            15000 + i,
            i
        ));
    }

    source.push_str(
        r#"
type ChEBIDrug = chebi:drug;

fn main() {
    let d: ChEBIDrug = chebi:15000;
}
"#,
    );

    TestHarness::new()
        .with_timing()
        .compile_str("many_alignments", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(15));
}

/// Test alignment chain performance with explicit direct alignment
/// Note: Transitive coercion not yet implemented, using direct alignment
#[test]
fn test_transitive_chain_performance() {
    let mut source = String::new();

    // Create 10 ontologies
    for i in 0..10 {
        source.push_str(&format!(
            "ontology onto{} from \"file://ontologies/onto{}.owl\";\n",
            i, i
        ));
    }

    // Chain alignments: onto0 ~ onto1 ~ onto2 ~ ... ~ onto9
    for i in 0..9 {
        source.push_str(&format!(
            "align onto{}:term ~ onto{}:term with distance 0.05;\n",
            i,
            i + 1
        ));
    }

    // Direct alignment from first to last (transitive not yet implemented)
    source.push_str("align onto0:term ~ onto9:term with distance 0.45;\n");

    source.push_str(
        r#"
type First = onto0:term;
type Last = onto9:term;

#[compat(threshold = 0.5)]  // 0.45 direct alignment
fn process(t: First) {}

fn main() {
    let last: Last = onto9:12345;
    process(last);  // Uses direct alignment
}
"#,
    );

    TestHarness::new()
        .with_timing()
        .compile_str("transitive_chain", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(20));
}

// ============================================================================
// Scaling Tests - Expressions
// ============================================================================

/// Test deeply nested expression performance
#[test]
fn test_nested_expression_performance() {
    let mut expr = "1".to_string();
    for _ in 0..100 {
        expr = format!("({} + 1)", expr);
    }

    let source = format!("fn main() {{ let x = {}; }}", expr);

    TestHarness::new()
        .with_timing()
        .compile_str("nested_expr", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(10));
}

/// Test long expression chain
#[test]
fn test_expression_chain_performance() {
    let mut source = String::from("fn main() { let x = 0");
    for i in 1..500 {
        source.push_str(&format!(" + {}", i));
    }
    source.push_str("; }\n");

    TestHarness::new()
        .with_timing()
        .compile_str("expr_chain", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(10));
}

// ============================================================================
// Scaling Tests - Control Flow
// ============================================================================

/// Test many if-else branches
#[test]
fn test_branch_scaling() {
    let mut source = String::from("fn dispatch(n: i32) -> i32 {\n");
    for i in 0..100 {
        source.push_str(&format!("    if n == {} {{ return {}; }}\n", i, i * 2));
    }
    source.push_str("    0\n}\nfn main() { dispatch(50); }\n");

    TestHarness::new()
        .with_timing()
        .compile_str("many_branches", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(10));
}

/// Test nested loops
#[test]
fn test_loop_nesting_performance() {
    let source = r#"
fn main() {
    let mut sum = 0;
    for i in 0..10 {
        for j in 0..10 {
            for k in 0..10 {
                for l in 0..10 {
                    sum += i * j * k * l;
                }
            }
        }
    }
}
"#;

    TestHarness::new()
        .with_timing()
        .compile_str("nested_loops", source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(5));
}

// ============================================================================
// Memory Tests
// ============================================================================

/// Test that large files don't cause OOM
#[test]
fn test_large_file_memory() {
    // Generate ~1MB of source code
    let mut source = String::with_capacity(1_000_000);
    source.push_str(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;
"#,
    );

    // Add many functions to bulk up the file
    for i in 0..1000 {
        source.push_str(&format!(
            r#"
fn func_{i}(d: Drug) -> Drug {{
    // Some computation
    let x = {i};
    let y = x + 1;
    d
}}
"#,
            i = i
        ));
    }

    source.push_str("fn main() {}\n");

    // Should complete without running out of memory
    TestHarness::new()
        .with_timing()
        .compile_str("large_file", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(60));
}

/// Test many string literals don't cause memory issues
#[test]
fn test_string_memory() {
    let mut source = String::from("fn main() {\n");
    for i in 0..1000 {
        let long_string = format!("String number {} with some padding text here", i);
        source.push_str(&format!("    let s{} = \"{}\";\n", i, long_string));
    }
    source.push_str("}\n");

    TestHarness::new()
        .with_timing()
        .compile_str("many_strings", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(30));
}

// ============================================================================
// Diagnostic Performance Tests
// ============================================================================

/// Test that error reporting doesn't slow down for many errors
#[test]
fn test_error_reporting_performance() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
"#,
    );

    // Generate many type errors
    for i in 0..50 {
        source.push_str(&format!("    let d{}: Drug = {};\n", i, i));
    }
    source.push_str("}\n");

    let result = TestHarness::new()
        .with_timing()
        .json_diagnostics()
        .compile_str("many_errors", &source);

    result
        .assert_failure()
        .assert_duration_under(Duration::from_secs(30));

    // Should report all errors
    assert!(
        result.error_count() >= 40,
        "Expected at least 40 errors, got {}",
        result.error_count()
    );
}

/// Test suggestion generation performance
#[test]
fn test_suggestion_generation_performance() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.3;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

fn main() {
"#,
    );

    // Generate many mismatches that need suggestions
    for i in 0..30 {
        source.push_str(&format!(
            r#"
    let db{}: DrugBankDrug = drugbank:DB{:05};
    let _c{}: ChEBIDrug = db{};  // Needs suggestion
"#,
            i, i, i, i
        ));
    }
    source.push_str("}\n");

    TestHarness::new()
        .with_timing()
        .json_diagnostics()
        .compile_str("many_suggestions", &source)
        .assert_duration_under(Duration::from_secs(45));
}

// ============================================================================
// Incremental Compilation Tests (if supported)
// ============================================================================

/// Test that recompiling unchanged code is fast
#[test]
fn test_recompile_unchanged() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:15365;
}
"#;

    let harness = TestHarness::new().with_timing();

    // First compile
    let result1 = harness.compile_str("recompile1", source);
    result1.assert_success();

    // Second compile of same code (might be cached)
    let result2 = harness.compile_str("recompile2", source);
    result2.assert_success();

    // Note: Without incremental compilation, times might be similar
    // With caching, second should be faster
    if let (Some(t1), Some(t2)) = (result1.duration(), result2.duration()) {
        // Just ensure neither is pathologically slow
        assert!(t1 < Duration::from_secs(30));
        assert!(t2 < Duration::from_secs(30));
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

/// Stress test: combine all scalability factors
#[test]
#[ignore] // Can be slow, run with --ignored
fn test_stress_combined() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";
ontology mondo from "https://purl.obolibrary.org/obo/mondo.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.1;
"#,
    );

    // Add many types
    for i in 0..50 {
        source.push_str(&format!("type DrugType{} = chebi:drug;\n", i));
    }

    // Add many structs
    for i in 0..20 {
        source.push_str(&format!(
            r#"
struct Model{} {{
    drug: DrugType0,
    phenotype: hp:phenotypic_abnormality,
    disease: mondo:disease,
    score: f64,
}}
"#,
            i
        ));
    }

    // Add many functions
    for i in 0..50 {
        source.push_str(&format!(
            r#"
fn analyze_{}(d: DrugType0, p: hp:phenotypic_abnormality) -> f64 {{
    let base = {}.0;
    base + 1.0
}}
"#,
            i, i
        ));
    }

    // Main function using everything
    source.push_str(
        r#"
fn main() {
    let drug: DrugType0 = chebi:15365;
    let phenotype: hp:phenotypic_abnormality = hp:0002315;
"#,
    );

    for i in 0..50 {
        source.push_str(&format!(
            "    let _r{} = analyze_{}(drug, phenotype);\n",
            i, i
        ));
    }

    source.push_str("}\n");

    TestHarness::new()
        .with_timing()
        .compile_str("stress_combined", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(120));
}

/// Stress test: maximum reasonable file size
#[test]
#[ignore] // Can be slow
fn test_stress_large_file() {
    let mut source = String::with_capacity(5_000_000);
    source.push_str("fn main() {\n");

    // Generate ~5MB of code
    for i in 0..50000 {
        source.push_str(&format!("    let var_{} = {};\n", i, i));
    }

    source.push_str("}\n");

    TestHarness::new()
        .with_timing()
        .compile_str("huge_file", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(180));
}
