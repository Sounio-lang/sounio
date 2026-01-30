// tests/e2e/golden.rs — Golden file (snapshot) tests
//
// These tests compare compiler output against known-good reference files.
// Run with UPDATE_GOLDEN=1 to regenerate golden files.
//
// Note: Golden tests are Linux-only because golden files contain Linux-specific
// paths and the normalization assumes /tmp/ paths. On macOS/Windows, these tests
// are automatically ignored.

use super::common::TestHarness;

// ============================================================================
// Error Message Golden Tests
// ============================================================================

/// Golden test for basic type mismatch error
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_type_mismatch_basic() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";

type Drug = chebi:drug;
type Process = go:biological_process;

fn need_drug(d: Drug) {}

fn main() {
    let p: Process = go:0006915;
    need_drug(p);
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("type_mismatch", source)
        .assert_stderr_matches_golden("errors/type_mismatch_basic.txt");
}

/// Golden test for semantic distance error
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_semantic_distance_error() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.4;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn strict_analysis(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    strict_analysis(db);
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("distance_error", source)
        .assert_stderr_matches_golden("errors/semantic_distance.txt");
}

/// Golden test for undefined ontology error
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_undefined_ontology() {
    let source = r#"
type Drug = undefined_ontology:drug;

fn main() {}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("undefined_onto", source)
        .assert_stderr_matches_golden("errors/undefined_ontology.txt");
}

/// Golden test for duplicate type definition
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_duplicate_type() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;
type Drug = chebi:small_molecule;

fn main() {}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("dup_type", source)
        .assert_stderr_matches_golden("errors/duplicate_type.txt");
}

// ============================================================================
// Warning Message Golden Tests
// ============================================================================

/// Golden test for unused import warning
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_unused_import() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";

type Drug = chebi:drug;

fn main() {
    let d: Drug = chebi:15365;
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .flag("--warn=unused-imports")
        .compile_str("unused_import", source)
        .assert_stderr_matches_golden("warnings/unused_import.txt");
}

/// Golden test for loose threshold warning
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_loose_threshold() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

#[compat(threshold = 0.95)]
fn very_loose(d: Drug) {}

fn main() {}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .flag("--warn=loose-threshold")
        .compile_str("loose_thresh", source)
        .assert_stderr_matches_golden("warnings/loose_threshold.txt");
}

// ============================================================================
// Multi-Error Golden Tests
// ============================================================================

/// Golden test for multiple errors in one file
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_multiple_errors() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let a: Drug = 1;
    let b: Drug = "string";
    let c: Drug = true;
    let d: Unknown = chebi:15365;
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("multi_error", source)
        .assert_stderr_matches_golden("errors/multiple_errors.txt");
}

/// Golden test for cascading errors
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_cascading_errors() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = undefined:drug;  // First error

fn use_drug(d: Drug) -> Drug {
    d
}

fn main() {
    let d: Drug = chebi:15365;
    use_drug(d);  // Cascading errors
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("cascade", source)
        .assert_stderr_matches_golden("errors/cascading.txt");
}

// ============================================================================
// Suggestion Golden Tests
// ============================================================================

/// Golden test for "did you mean" suggestions
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_did_you_mean() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Durg = chebi:drug;  // Typo

fn main() {
    let d: Drug = chebi:15365;  // Should suggest Durg
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("did_you_mean", source)
        .assert_stderr_matches_golden("suggestions/did_you_mean.txt");
}

/// Golden test for threshold adjustment suggestion
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_threshold_suggestion() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.35;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.3)]
fn analyze(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    analyze(db);
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("threshold_suggest", source)
        .assert_stderr_matches_golden("suggestions/threshold_adjustment.txt");
}

/// Golden test for alignment suggestion
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_alignment_suggestion() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// No alignment declared

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

fn analyze(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    analyze(db);  // Should suggest declaring alignment
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("align_suggest", source)
        .assert_stderr_matches_golden("suggestions/alignment.txt");
}

// ============================================================================
// Complex Scenario Golden Tests
// ============================================================================

/// Golden test for pharmacology type mismatch
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_pharma_mismatch() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";

type Drug = chebi:drug;
type Phenotype = hp:phenotypic_abnormality;

struct DrugIndication {
    drug: Drug,
    treats: Phenotype,
}

fn create_indication(d: Drug, p: Phenotype) -> DrugIndication {
    DrugIndication {
        drug: p,    // Swapped! Should be d
        treats: d,  // Swapped! Should be p
    }
}

fn main() {}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("pharma_mismatch", source)
        .assert_stderr_matches_golden("scenarios/pharma_field_swap.txt");
}

/// Golden test for cross-ontology coercion chain
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_coercion_chain() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";
ontology rxnorm from "file://ontologies/rxnorm.owl";

align chebi:drug ~ drugbank:drug with distance 0.15;
align drugbank:drug ~ rxnorm:clinical_drug with distance 0.15;

type ChEBIDrug = chebi:drug;
type RxNormDrug = rxnorm:clinical_drug;

#[compat(threshold = 0.25)]  // 0.15 + 0.15 = 0.30, which exceeds 0.25
fn process(d: ChEBIDrug) {}

fn main() {
    let rx: RxNormDrug = rxnorm:1191;
    process(rx);  // Should show transitive distance
}
"#;

    TestHarness::new()
        .flag("--error-format=human")
        .compile_str("coercion_chain", source)
        .assert_stderr_matches_golden("scenarios/transitive_distance.txt");
}

// ============================================================================
// JSON Output Golden Tests
// ============================================================================

/// Golden test for JSON error format
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_json_error() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn main() {
    let d: Drug = 42;
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("json_error", source)
        .assert_stderr_matches_golden("json/error_format.json");
}

/// Golden test for JSON with semantic distance
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_json_distance() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.3;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn analyze(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00945;
    analyze(db);
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("json_distance", source)
        .assert_stderr_matches_golden("json/semantic_distance.json");
}

// ============================================================================
// Help Text Golden Tests
// ============================================================================

/// Golden test for help output
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_help_output() {
    let harness = TestHarness::new();
    let result = harness.flag("--help").compile_str("help", "");

    result.assert_stdout_matches_golden("help/main_help.txt");
}

/// Golden test for version output
#[test]
#[cfg_attr(not(target_os = "linux"), ignore)]
fn golden_version_output() {
    let harness = TestHarness::new();
    let result = harness.flag("--version").compile_str("version", "");

    // Version might change, so we just check it doesn't crash
    // and produces some output
    let stdout = result.stdout();
    assert!(!stdout.is_empty() || !result.stderr().is_empty());
}

// ============================================================================
// Utility for Generating Initial Golden Files
// ============================================================================

/// Helper test to generate all golden files at once
/// Run with: UPDATE_GOLDEN=1 cargo test generate_all_golden -- --ignored
#[test]
#[ignore]
fn generate_all_golden() {
    // This will run all golden tests in update mode
    // Individual tests will write their golden files
    println!("Run individual golden tests with UPDATE_GOLDEN=1 to generate golden files");
}
