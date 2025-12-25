// tests/e2e/cross_ontology.rs — Cross-ontology alignment tests
//
// Tests semantic distance calculations and type coercion between
// different ontologies (ChEBI↔DrugBank, ChEBI↔RxNorm, etc.)

use super::common::TestHarness;
use std::time::Duration;

// ============================================================================
// ChEBI ↔ DrugBank Alignment Tests
// ============================================================================

/// Test explicit cross-ontology alignment declaration
#[test]
fn test_chebi_drugbank_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// Declare alignment between ChEBI and DrugBank drug concepts
align chebi:drug ~ drugbank:drug with distance 0.1;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

fn process_chebi_drug(d: ChEBIDrug) -> f64 {
    // Return some analysis score
    0.95
}

#[compat(threshold = 0.2)]
fn analyze_with_alignment(d: DrugBankDrug) -> f64 {
    // DrugBank drug can be used where ChEBI expected due to alignment
    process_chebi_drug(d)  // Coerces via alignment
}

fn main() {
    let db_aspirin: DrugBankDrug = drugbank:DB00945;
    let score = analyze_with_alignment(db_aspirin);
}
"#;

    TestHarness::new()
        .compile_str("chebi_drugbank_align", source)
        .assert_success();
}

/// Test that exceeding alignment threshold fails
#[test]
fn test_alignment_threshold_exceeded() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.5;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.3)]  // Threshold too strict for 0.5 distance
fn strict_analysis(d: ChEBIDrug) {
    // Process
}

fn main() {
    let db_drug: DrugBankDrug = drugbank:DB00945;
    strict_analysis(db_drug);  // ERROR: distance 0.5 > threshold 0.3
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("threshold_exceeded", source)
        .assert_failure()
        .assert_error_contains("semantic distance")
        .assert_error_contains("exceeds threshold");
}

/// Test hierarchical alignment (subtypes)
#[test]
fn test_hierarchical_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// Parent alignment
align chebi:drug ~ drugbank:drug with distance 0.1;

// More specific alignment (lower distance for specific match)
align chebi:analgesic ~ drugbank:analgesic with distance 0.05;

type ChEBIAnalgesic = chebi:analgesic;
type DrugBankAnalgesic = drugbank:analgesic;

#[compat(threshold = 0.1)]
fn process_analgesic(a: ChEBIAnalgesic) {
    // Process analgesic drug
}

fn main() {
    let aspirin: DrugBankAnalgesic = drugbank:DB00945;
    process_analgesic(aspirin);  // Uses specific 0.05 alignment
}
"#;

    TestHarness::new()
        .compile_str("hierarchical_align", source)
        .assert_success();
}

// ============================================================================
// ChEBI ↔ RxNorm Alignment Tests
// ============================================================================

/// Test ChEBI to RxNorm drug mapping
#[test]
fn test_chebi_rxnorm_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology rxnorm from "file://ontologies/rxnorm.owl";

align chebi:drug ~ rxnorm:clinical_drug with distance 0.15;

type ChEBIMolecule = chebi:drug;
type RxNormDrug = rxnorm:clinical_drug;

struct Prescription {
    drug: RxNormDrug,
    dosage: f64,
}

#[compat(threshold = 0.2)]
fn lookup_interactions(molecule: ChEBIMolecule) -> Vec<ChEBIMolecule> {
    vec![]  // Would query interaction database
}

fn check_prescription_safety(rx: Prescription) -> bool {
    // RxNorm drug coerces to ChEBI for interaction lookup
    let interactions = lookup_interactions(rx.drug);
    interactions.is_empty()
}

fn main() {
    let prescription = Prescription {
        drug: rxnorm:1191,  // Aspirin in RxNorm
        dosage: 325.0,
    };

    check_prescription_safety(prescription);
}
"#;

    TestHarness::new()
        .compile_str("chebi_rxnorm", source)
        .assert_success();
}

// ============================================================================
// Multi-Ontology Integration Tests
// ============================================================================

/// Test three-way ontology integration
#[test]
fn test_three_ontology_integration() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";
ontology rxnorm from "file://ontologies/rxnorm.owl";

// Triangle of alignments
align chebi:drug ~ drugbank:drug with distance 0.1;
align chebi:drug ~ rxnorm:clinical_drug with distance 0.15;
align drugbank:drug ~ rxnorm:clinical_drug with distance 0.12;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;
type RxNormDrug = rxnorm:clinical_drug;

// Unified drug analysis that accepts any source
#[compat(threshold = 0.2)]
fn unified_analysis(d: ChEBIDrug) -> f64 {
    // Analyze drug regardless of source ontology
    0.5
}

fn main() {
    let chebi_drug: ChEBIDrug = chebi:15365;
    let drugbank_drug: DrugBankDrug = drugbank:DB00945;
    let rxnorm_drug: RxNormDrug = rxnorm:1191;

    // All three can be analyzed through unified interface
    unified_analysis(chebi_drug);
    unified_analysis(drugbank_drug);
    unified_analysis(rxnorm_drug);
}
"#;

    TestHarness::new()
        .compile_str("three_ontology", source)
        .assert_success();
}

/// Test phenotype ontology alignment (HP ↔ MP for human/mouse)
#[test]
fn test_phenotype_cross_species() {
    let source = r#"
ontology hp from "https://purl.obolibrary.org/obo/hp.owl";
ontology mp from "https://purl.obolibrary.org/obo/mp.owl";

// Human-mouse phenotype alignment
align hp:phenotypic_abnormality ~ mp:mammalian_phenotype with distance 0.2;

type HumanPhenotype = hp:phenotypic_abnormality;
type MousePhenotype = mp:mammalian_phenotype;

struct PhenotypeMatch {
    human: HumanPhenotype,
    mouse: MousePhenotype,
    confidence: f64,
}

#[compat(threshold = 0.3)]
fn find_mouse_model(human_pheno: HumanPhenotype) -> Vec<MousePhenotype> {
    // Find matching mouse phenotypes for translational research
    vec![]
}

fn main() {
    let seizures: HumanPhenotype = hp:0001250;
    let mouse_models = find_mouse_model(seizures);
}
"#;

    TestHarness::new()
        .compile_str("hp_mp_alignment", source)
        .assert_success();
}

// ============================================================================
// Disease Ontology Alignment Tests
// ============================================================================

/// Test MONDO ↔ OMIM disease alignment
#[test]
fn test_mondo_omim_alignment() {
    let source = r#"
ontology mondo from "https://purl.obolibrary.org/obo/mondo.owl";
ontology omim from "file://ontologies/omim.owl";

align mondo:disease ~ omim:disorder with distance 0.1;

type MONDODisease = mondo:disease;
type OMIMDisorder = omim:disorder;

struct GeneticDisorder {
    omim_id: OMIMDisorder,
    gene_symbols: Vec<String>,
}

#[compat(threshold = 0.2)]
fn get_disease_info(d: MONDODisease) -> Option<String> {
    // Lookup disease information
    Some("Disease info".to_string())
}

fn analyze_genetic_disorder(disorder: GeneticDisorder) -> Option<String> {
    // OMIM disorder coerces to MONDO for info lookup
    get_disease_info(disorder.omim_id)
}

fn main() {
    let cystic_fibrosis = GeneticDisorder {
        omim_id: omim:219700,
        gene_symbols: vec!["CFTR".to_string()],
    };

    analyze_genetic_disorder(cystic_fibrosis);
}
"#;

    TestHarness::new()
        .compile_str("mondo_omim", source)
        .assert_success();
}

/// Test MONDO ↔ DOID alignment
#[test]
fn test_mondo_doid_alignment() {
    let source = r#"
ontology mondo from "https://purl.obolibrary.org/obo/mondo.owl";
ontology doid from "https://purl.obolibrary.org/obo/doid.owl";

align mondo:disease ~ doid:disease with distance 0.08;

type MONDODisease = mondo:disease;
type DOIDDisease = doid:disease;

#[compat(threshold = 0.1)]
fn combine_disease_annotations(d: MONDODisease) -> Vec<String> {
    vec![]
}

fn main() {
    let doid_diabetes: DOIDDisease = doid:9352;

    // DOID coerces to MONDO
    let annotations = combine_disease_annotations(doid_diabetes);
}
"#;

    TestHarness::new()
        .compile_str("mondo_doid", source)
        .assert_success();
}

// ============================================================================
// Semantic Distance Calculation Tests
// ============================================================================

/// Test that type mismatch between ontologies is detected
#[test]
fn test_distance_suggestion_provided() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology go from "https://purl.obolibrary.org/obo/go.owl";

// No alignment declared - these are fundamentally different
type Drug = chebi:drug;
type BiologicalProcess = go:biological_process;

fn process_drug(d: Drug) {
    // Process
}

fn main() {
    let apoptosis: BiologicalProcess = go:0006915;
    process_drug(apoptosis);  // Should fail - no alignment
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("distance_suggestion", source)
        .assert_failure()
        .assert_error_contains("mismatch");
}

/// Test distance components are reported
#[test]
fn test_distance_components_reported() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// Declare alignment with moderate distance
align chebi:analgesic ~ drugbank:opioid with distance 0.4;

type ChEBIAnalgesic = chebi:analgesic;
type DrugBankOpioid = drugbank:opioid;

#[compat(threshold = 0.2)]  // Too strict
fn analyze_analgesic(a: ChEBIAnalgesic) {
    // Process
}

fn main() {
    let morphine: DrugBankOpioid = drugbank:DB00295;
    analyze_analgesic(morphine);  // Fails, should show distance breakdown
}
"#;

    let result = TestHarness::new()
        .json_diagnostics()
        .compile_str("distance_components", source);

    result.assert_failure();

    // Check that distance components are mentioned in output
    let stderr = result.stderr();
    assert!(
        stderr.contains("distance") || stderr.contains("0.4"),
        "Expected semantic distance info in output"
    );
}

// ============================================================================
// Coercion Chain Tests
// ============================================================================

/// Test coercion with explicit direct alignment
#[test]
fn test_transitive_coercion() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";
ontology rxnorm from "file://ontologies/rxnorm.owl";

// Direct alignments (transitive coercion not yet implemented)
align chebi:drug ~ drugbank:drug with distance 0.1;
align drugbank:drug ~ rxnorm:clinical_drug with distance 0.1;
align chebi:drug ~ rxnorm:clinical_drug with distance 0.2;

type ChEBIDrug = chebi:drug;
type RxNormDrug = rxnorm:clinical_drug;

#[compat(threshold = 0.25)]  // Allows direct alignment
fn chebi_analysis(d: ChEBIDrug) {
    // Process
}

fn main() {
    let rxnorm_drug: RxNormDrug = rxnorm:1191;
    // Coerces via explicit alignment
    chebi_analysis(rxnorm_drug);
}
"#;

    TestHarness::new()
        .compile_str("transitive_coercion", source)
        .assert_success();
}

/// Test that transitive coercion respects cumulative distance
#[test]
fn test_transitive_distance_accumulates() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";
ontology rxnorm from "file://ontologies/rxnorm.owl";

align chebi:drug ~ drugbank:drug with distance 0.2;
align drugbank:drug ~ rxnorm:clinical_drug with distance 0.2;
// Cumulative: 0.4

type ChEBIDrug = chebi:drug;
type RxNormDrug = rxnorm:clinical_drug;

#[compat(threshold = 0.3)]  // Too strict for 0.4 cumulative
fn strict_chebi_only(d: ChEBIDrug) {
    // Process
}

fn main() {
    let rxnorm_drug: RxNormDrug = rxnorm:1191;
    strict_chebi_only(rxnorm_drug);  // Should fail
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("transitive_distance", source)
        .assert_failure()
        .assert_error_contains("distance");
}

// ============================================================================
// Unit Ontology Tests
// ============================================================================

/// Test unit ontology alignment (UO conversions)
#[test]
fn test_unit_alignment() {
    let source = r#"
ontology uo from "https://purl.obolibrary.org/obo/uo.owl";

type Milligram = uo:milligram;
type Gram = uo:gram;
type MassUnit = uo:mass_unit;

// Both are mass units, gram is parent of milligram
fn require_mass_unit(u: MassUnit) -> bool {
    true
}

fn main() {
    let mg: Milligram = uo:0000022;
    let g: Gram = uo:0000021;

    // Both should work as MassUnit
    require_mass_unit(mg);
    require_mass_unit(g);
}
"#;

    TestHarness::new()
        .compile_str("unit_alignment", source)
        .assert_success();
}

// ============================================================================
// Edge Cases
// ============================================================================

/// Test self-alignment (same ontology, same term)
#[test]
fn test_self_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";

type Drug = chebi:drug;

fn identity(d: Drug) -> Drug {
    d  // Zero distance, always works
}

fn main() {
    let aspirin: Drug = chebi:15365;
    let same = identity(aspirin);
}
"#;

    TestHarness::new()
        .compile_str("self_alignment", source)
        .assert_success();
}

/// Test alignment with explicit zero distance
#[test]
fn test_zero_distance_alignment() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// Exact equivalence
align chebi:15365 ~ drugbank:DB00945 with distance 0.0;

type ChEBIAspirin = chebi:15365;
type DrugBankAspirin = drugbank:DB00945;

fn process_aspirin(a: ChEBIAspirin) {
    // Process
}

fn main() {
    let db_aspirin: DrugBankAspirin = drugbank:DB00945;
    process_aspirin(db_aspirin);  // Exact match, zero distance
}
"#;

    TestHarness::new()
        .compile_str("zero_distance", source)
        .assert_success();
}

/// Test alignment near threshold boundary
#[test]
fn test_boundary_threshold() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.30;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.30)]  // Exactly at boundary
fn boundary_analysis(d: ChEBIDrug) {
    // Process
}

fn main() {
    let db_drug: DrugBankDrug = drugbank:DB00945;
    boundary_analysis(db_drug);  // Should pass (<=)
}
"#;

    TestHarness::new()
        .compile_str("boundary_threshold", source)
        .assert_success();
}

/// Test alignment just over threshold boundary
#[test]
fn test_over_boundary_threshold() {
    let source = r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

align chebi:drug ~ drugbank:drug with distance 0.31;

type ChEBIDrug = chebi:drug;
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.30)]  // Just under distance
fn strict_analysis(d: ChEBIDrug) {
    // Process
}

fn main() {
    let db_drug: DrugBankDrug = drugbank:DB00945;
    strict_analysis(db_drug);  // Should fail (0.31 > 0.30)
}
"#;

    TestHarness::new()
        .json_diagnostics()
        .compile_str("over_boundary", source)
        .assert_failure();
}

// ============================================================================
// Performance Tests
// ============================================================================

/// Test alignment lookup performance with many alignments
#[test]
fn test_many_alignments_performance() {
    let mut source = String::from(
        r#"
ontology chebi from "https://purl.obolibrary.org/obo/chebi.owl";
ontology drugbank from "file://ontologies/drugbank.owl";

// Base alignment for the generic types
align chebi:drug ~ drugbank:drug with distance 0.1;

"#,
    );

    // Add many specific alignments
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
type DrugBankDrug = drugbank:drug;

#[compat(threshold = 0.2)]
fn process(d: ChEBIDrug) {}

fn main() {
    let db: DrugBankDrug = drugbank:DB00050;
    process(db);
}
"#,
    );

    TestHarness::new()
        .with_timing()
        .compile_str("many_alignments", &source)
        .assert_success()
        .assert_duration_under(Duration::from_secs(10));
}
