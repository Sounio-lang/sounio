//! Medical Ontology Integration Demo
//!
//! This example demonstrates how to use Sounio' ontology integration
//! to build a medical prescription validation system using real biomedical
//! ontologies (ChEBI, DOID, SNOMED, FHIR).
//!
//! Run with: cargo run --example medical_ontology_demo --features ontology

use sounio::ontology::{
    OntologyResolver, ResolverConfig, SubsumptionResult,
    loader::{OntologyId, IRI},
};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("=== Sounio Medical Ontology Demo ===\n");

    // Initialize resolver with offline mode (no BioPortal API needed)
    let config = ResolverConfig::default()
        .with_data_dir("./ontology_cache")
        .offline();

    let mut resolver = OntologyResolver::new(config)?;

    // Demo 1: Resolve pharmaceutical terms
    demo_pharmaceutical_resolution(&mut resolver)?;

    // Demo 2: Check drug classification
    demo_drug_classification(&mut resolver)?;

    // Demo 3: Disease hierarchy
    demo_disease_hierarchy(&mut resolver)?;

    // Demo 4: FHIR resource types
    demo_fhir_resources(&mut resolver)?;

    // Demo 5: Performance statistics
    demo_statistics(&resolver);

    Ok(())
}

/// Demo 1: Resolve pharmaceutical terms from ChEBI
fn demo_pharmaceutical_resolution(resolver: &mut OntologyResolver) -> Result<(), Box<dyn Error>> {
    println!("📋 Demo 1: Pharmaceutical Term Resolution\n");

    let drugs = vec![
        ("CHEBI:15365", "Aspirin"),
        ("CHEBI:5855", "Ibuprofen"),
        ("CHEBI:6801", "Morphine"),
        ("CHEBI:2618", "Amoxicillin"),
    ];

    for (curie, expected_name) in drugs {
        match resolver.resolve(curie) {
            Ok(term) => {
                println!("✓ {} ({})", curie, expected_name);
                println!("  Label: {}", term.label.unwrap_or_default());
                if let Some(def) = &term.definition {
                    let short_def = def.chars().take(80).collect::<String>();
                    println!("  Definition: {}...", short_def);
                }
                println!("  Layer: {:?}", term.layer);
                println!("  Superclasses: {}", term.superclasses.len());
                println!();
            }
            Err(e) => {
                println!("✗ {} - Not found ({})", curie, e);
                println!("  Note: May require L3 domain ontology database or L4 network access");
                println!();
            }
        }
    }

    Ok(())
}

/// Demo 2: Check if drugs are properly classified
fn demo_drug_classification(resolver: &mut OntologyResolver) -> Result<(), Box<dyn Error>> {
    println!("🔍 Demo 2: Drug Classification via Subsumption\n");

    let test_cases = vec![
        ("CHEBI:15365", "CHEBI:35475", "Aspirin is-a NSAID"),
        ("CHEBI:15365", "CHEBI:23888", "Aspirin is-a drug"),
        ("CHEBI:5855", "CHEBI:35475", "Ibuprofen is-a NSAID"),
        ("CHEBI:6801", "CHEBI:23888", "Morphine is-a drug"),
        ("CHEBI:15365", "GO:0008150", "Aspirin is NOT a biological process (should fail)"),
    ];

    for (child, parent, description) in test_cases {
        match resolver.is_subclass_of(child, parent) {
            Ok(result) => {
                let symbol = match result {
                    SubsumptionResult::IsSubclass => "✓",
                    SubsumptionResult::Equivalent => "=",
                    SubsumptionResult::NotSubclass => "✗",
                    SubsumptionResult::Unknown => "?",
                };
                println!("{} {}", symbol, description);
                println!("  Result: {:?}", result);
            }
            Err(e) => {
                println!("? {} (Error: {})", description, e);
            }
        }
        println!();
    }

    Ok(())
}

/// Demo 3: Explore disease hierarchy
fn demo_disease_hierarchy(resolver: &mut OntologyResolver) -> Result<(), Box<dyn Error>> {
    println!("🏥 Demo 3: Disease Hierarchy Exploration\n");

    let diseases = vec![
        "DOID:162",   // Cancer
        "DOID:3910",  // Lung cancer
        "DOID:9351",  // Diabetes mellitus
    ];

    for disease_id in diseases {
        println!("Disease: {}", disease_id);

        match resolver.resolve(disease_id) {
            Ok(disease) => {
                println!("  Label: {}", disease.label.unwrap_or_default());

                if let Some(def) = &disease.definition {
                    let short_def = def.chars().take(100).collect::<String>();
                    println!("  Definition: {}...", short_def);
                }

                // Get ancestors
                match resolver.get_ancestors(disease_id) {
                    Ok(ancestors) => {
                        println!("  Ancestors ({}):", ancestors.len());
                        for (i, ancestor) in ancestors.iter().take(5).enumerate() {
                            println!("    {}. {}", i + 1, ancestor);
                        }
                        if ancestors.len() > 5 {
                            println!("    ... and {} more", ancestors.len() - 5);
                        }
                    }
                    Err(e) => {
                        println!("  Ancestors: Error ({})", e);
                    }
                }
            }
            Err(e) => {
                println!("  Not found: {}", e);
                println!("  Note: Requires L3 DOID database or L4 BioPortal access");
            }
        }
        println!();
    }

    Ok(())
}

/// Demo 4: FHIR R5 resource types (Foundation layer)
fn demo_fhir_resources(resolver: &mut OntologyResolver) -> Result<(), Box<dyn Error>> {
    println!("💉 Demo 4: FHIR R5 Healthcare Resources (L2 Foundation)\n");

    let fhir_resources = vec![
        "FHIR:Patient",
        "FHIR:Observation",
        "FHIR:Medication",
        "FHIR:Condition",
        "FHIR:Procedure",
        "FHIR:DiagnosticReport",
    ];

    for resource in fhir_resources {
        match resolver.resolve(resource) {
            Ok(term) => {
                println!("✓ {}", resource);
                println!("  Label: {}", term.label.unwrap_or_default());
                println!("  Layer: {:?}", term.layer);
            }
            Err(_) => {
                println!("✗ {} - Not yet in foundation layer", resource);
                println!("  Note: FHIR foundation ontology may need population");
            }
        }
        println!();
    }

    Ok(())
}

/// Demo 5: Show resolver statistics
fn demo_statistics(resolver: &OntologyResolver) {
    println!("📊 Demo 5: Resolver Statistics\n");

    let stats = resolver.stats();
    println!("Ontology Resolution Stats:");
    println!("  L1 (Primitive) hits: {}", stats.primitive_hits);
    println!("  L2 (Foundation) hits: {}", stats.foundation_hits);
    println!("  L3 (Domain) hits: {}", stats.domain_hits);
    println!("  L4 (Federated) hits: {}", stats.federated_hits);
    println!("  Total resolutions: {}", stats.total_resolutions());
    println!();

    let cache_stats = resolver.cache_stats();
    println!("Cache Stats:");
    println!("  Total hits: {}", cache_stats.total_hits());
    println!("  Hot cache hits: {}", cache_stats.hot_hits);
    println!("  Warm cache hits: {}", cache_stats.warm_hits);
    println!("  Cold cache hits: {}", cache_stats.cold_hits);
    println!("  Cache misses: {}", cache_stats.misses);
    println!("  Hit rate: {:.2}%", cache_stats.hit_rate() * 100.0);
    println!();

    println!("Subsumption Checks: {}", stats.subsumption_checks);
    println!("Mapping Lookups: {}", stats.mapping_lookups);
    println!();
}

/// Advanced Demo: Prescription Validation
#[allow(dead_code)]
fn validate_prescription(
    resolver: &mut OntologyResolver,
    drug_curie: &str,
    condition_curie: &str,
) -> Result<bool, Box<dyn Error>> {
    println!("🔒 Validating Prescription:");
    println!("  Drug: {}", drug_curie);
    println!("  Condition: {}", condition_curie);

    // Resolve both terms
    let drug = resolver.resolve(drug_curie)?;
    let condition = resolver.resolve(condition_curie)?;

    println!("  Drug label: {}", drug.label.unwrap_or_default());
    println!("  Condition label: {}", condition.label.unwrap_or_default());

    // Verify drug is actually a drug (CHEBI:23888)
    let is_drug = resolver.is_subclass_of(drug_curie, "CHEBI:23888")?;

    if !matches!(is_drug, SubsumptionResult::IsSubclass) {
        println!("  ✗ Invalid: {} is not a drug", drug_curie);
        return Ok(false);
    }

    // Verify condition is actually a disease (DOID:4)
    let is_disease = resolver.is_subclass_of(condition_curie, "DOID:4")?;

    if !matches!(is_disease, SubsumptionResult::IsSubclass) {
        println!("  ✗ Invalid: {} is not a disease", condition_curie);
        return Ok(false);
    }

    println!("  ✓ Valid prescription");
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolver_initialization() {
        let config = ResolverConfig::default().offline();
        let resolver = OntologyResolver::new(config);
        assert!(resolver.is_ok());
    }

    #[test]
    fn test_primitive_resolution() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        // BFO:0000001 is "entity" (L1 Primitive)
        let entity = resolver.resolve("BFO:0000001");
        assert!(entity.is_ok());

        let entity = entity.unwrap();
        assert_eq!(entity.label, Some("entity".to_string()));
    }

    #[test]
    fn test_subsumption_same_term() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        let result = resolver.is_subclass_of("BFO:0000001", "BFO:0000001").unwrap();
        assert!(matches!(result, SubsumptionResult::Equivalent));
    }

    #[test]
    fn test_bfo_hierarchy() {
        let mut resolver = OntologyResolver::default_resolver().unwrap();

        // BFO:0000027 (object) is-a BFO:0000001 (entity)
        let result = resolver.is_subclass_of("BFO:0000027", "BFO:0000001").unwrap();
        assert!(matches!(result, SubsumptionResult::IsSubclass));
    }
}
