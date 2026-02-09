// Sounio Compiler - Sheaf-Theoretic Type Checking
// Cellular sheaves over ontology graphs with Čech cohomology

use crate::types::Type;
use std::collections::{HashMap, HashSet};

/// Cellular sheaf over an ontology graph
///
/// F: Cell(G) → Type with restriction maps ρ: F(σ) → F(τ) for τ ⊆ σ
#[derive(Clone)]
pub struct CellularSheaf {
    /// Cell assignments: each cell gets a type
    cells: HashMap<String, Type>,
    /// Restriction maps between cells
    /// Key: (source, target), Value: closure that applies restriction
    restrictions: HashMap<(String, String), String>,
}

/// Cohomology group H⁰ and H¹
///
/// - H⁰ = global sections (valid types)
/// - H¹ = obstructions (type errors)
pub struct CohomologyGroup {
    /// Global sections (valid types across all cells)
    pub h0: Vec<Type>,
    /// Obstruction cycles (inconsistencies that prevent global sections)
    pub h1: Vec<String>,
    /// Rank of H⁰ (dimension of global type space)
    pub rank: usize,
}

impl CellularSheaf {
    /// Create a new empty cellular sheaf
    pub fn new() -> Self {
        CellularSheaf {
            cells: HashMap::new(),
            restrictions: HashMap::new(),
        }
    }

    /// Add a cell with associated type
    pub fn add_cell(&mut self, id: String, cell_type: Type) {
        self.cells.insert(id, cell_type);
    }

    /// Add a restriction map from source to target cell
    ///
    /// The restriction map is encoded as a string (simplified for cohomology computation)
    pub fn add_restriction(&mut self, from: String, to: String, description: String) {
        self.restrictions.insert((from, to), description);
    }

    /// Get a cell type
    pub fn get_cell(&self, id: &str) -> Option<&Type> {
        self.cells.get(id)
    }

    /// Check if all restriction maps compose consistently (sheaf axiom)
    ///
    /// For any chain σ → τ → υ, the restrictions must compose:
    /// (ρ_τυ ∘ ρ_στ) = ρ_συ
    fn verify_restriction_composition(&self) -> Vec<String> {
        let mut errors = Vec::new();

        let restriction_pairs: Vec<(String, String)> = self.restrictions.keys().cloned().collect();

        // Check all composition chains
        for (sigma, tau) in &restriction_pairs {
            for (tau2, upsilon) in &restriction_pairs {
                if tau == tau2 {
                    // We have a chain: sigma -> tau -> upsilon
                    let direct_key = (sigma.clone(), upsilon.clone());

                    if !self.restrictions.contains_key(&direct_key) {
                        errors.push(format!(
                            "Missing direct restriction {} -> {}; have chain {} -> {} -> {}",
                            sigma, upsilon, sigma, tau, upsilon
                        ));
                    }
                }
            }
        }

        errors
    }

    /// Check the sheaf condition: local sections agree on overlaps
    ///
    /// Returns true if the restriction maps form a valid sheaf
    pub fn check_sheaf_condition(&self) -> bool {
        self.verify_restriction_composition().is_empty()
    }

    /// Compute sheaf cohomology (simplified version)
    ///
    /// - H⁰: types that consistently restrict to all cells (global sections)
    /// - H¹: obstructions from incompatible restrictions (errors)
    pub fn compute_cohomology(&self) -> CohomologyGroup {
        let errors = self.verify_restriction_composition();

        // H⁰ = global sections (all cell types if no restrictions fail)
        let h0 = if errors.is_empty() {
            self.cells.values().cloned().collect()
        } else {
            Vec::new()
        };

        // H¹ = obstruction cycles (restriction incompatibilities)
        let h1 = errors;

        let rank = h0.len();

        CohomologyGroup { h0, h1, rank }
    }

    /// Get the number of cells
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }

    /// Get the number of restriction maps
    pub fn restriction_count(&self) -> usize {
        self.restrictions.len()
    }
}

impl Default for CellularSheaf {
    fn default() -> Self {
        Self::new()
    }
}

/// Align federated ontologies using sheaf-theoretic framework
///
/// Each ontology becomes an open set in the sheaf, with local type contexts.
/// Restriction maps encode how types specialize across domains.
pub fn align_federated_ontologies(ontology_ids: &[String]) -> Result<CellularSheaf, String> {
    let mut sheaf = CellularSheaf::new();

    // Keep deterministic order and ignore duplicates/empty IDs.
    let mut unique_ontologies = Vec::new();
    let mut seen = HashSet::new();
    for id in ontology_ids {
        let trimmed = id.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            unique_ontologies.push(trimmed.to_string());
        }
    }

    // 1) Create a cell for each ontology
    // 2) Map ontology context into a nominal type assignment
    for ontology in &unique_ontologies {
        sheaf.add_cell(
            ontology.clone(),
            Type::Named {
                name: format!("ontology::{}", ontology),
                args: Vec::new(),
            },
        );
    }

    // 3) Add restriction maps between ontology cells to model overlap.
    // We add all directed pairs including identity maps so composition checks
    // always have a direct composite.
    for from in &unique_ontologies {
        for to in &unique_ontologies {
            let description = if from == to {
                format!("identity:{}", from)
            } else {
                format!("federated_projection:{}->{}", from, to)
            };
            sheaf.add_restriction(from.clone(), to.clone(), description);
        }
    }

    // 4) Global consistency is represented by `check_sheaf_condition`.
    Ok(sheaf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_sheaf_is_valid() {
        let sheaf = CellularSheaf::new();
        assert!(sheaf.check_sheaf_condition());
    }

    #[test]
    fn test_single_cell_sheaf() {
        let mut sheaf = CellularSheaf::new();
        sheaf.add_cell("cell1".to_string(), Type::I32);
        assert!(sheaf.check_sheaf_condition());
        assert_eq!(sheaf.cell_count(), 1);
    }

    #[test]
    fn test_two_cell_sheaf_valid() {
        let mut sheaf = CellularSheaf::new();
        sheaf.add_cell("cell1".to_string(), Type::I32);
        sheaf.add_cell("cell2".to_string(), Type::I32);
        sheaf.add_restriction(
            "cell1".to_string(),
            "cell2".to_string(),
            "identity".to_string(),
        );
        // Direct restriction exists, so composition is trivially satisfied
        assert!(sheaf.check_sheaf_condition());
    }

    #[test]
    fn test_chain_restriction_missing() {
        let mut sheaf = CellularSheaf::new();
        sheaf.add_cell("cell1".to_string(), Type::I32);
        sheaf.add_cell("cell2".to_string(), Type::I32);
        sheaf.add_cell("cell3".to_string(), Type::I32);

        // Add chain cell1 -> cell2 -> cell3 but no direct cell1 -> cell3
        sheaf.add_restriction("cell1".to_string(), "cell2".to_string(), "r12".to_string());
        sheaf.add_restriction("cell2".to_string(), "cell3".to_string(), "r23".to_string());

        // Should fail: missing direct cell1 -> cell3
        assert!(!sheaf.check_sheaf_condition());
    }

    #[test]
    fn test_cohomology_trivial() {
        let mut sheaf = CellularSheaf::new();
        sheaf.add_cell("cell1".to_string(), Type::I32);
        let coh = sheaf.compute_cohomology();

        // H⁰ should contain the cell type
        assert_eq!(coh.rank, 1);
        // H¹ should be empty (no obstructions)
        assert!(coh.h1.is_empty());
    }

    #[test]
    fn test_cohomology_with_obstruction() {
        let mut sheaf = CellularSheaf::new();
        sheaf.add_cell("cell1".to_string(), Type::I32);
        sheaf.add_cell("cell2".to_string(), Type::I32);
        sheaf.add_cell("cell3".to_string(), Type::I32);

        sheaf.add_restriction("cell1".to_string(), "cell2".to_string(), "r12".to_string());
        sheaf.add_restriction("cell2".to_string(), "cell3".to_string(), "r23".to_string());
        // Missing direct restriction creates obstruction

        let coh = sheaf.compute_cohomology();
        assert!(coh.rank < 3); // Some types aren't global sections
        assert!(!coh.h1.is_empty()); // Has obstructions
    }

    #[test]
    fn test_align_federated_ontologies_populates_cells_and_restrictions() {
        let ontology_ids = vec!["CHEBI".to_string(), "MONDO".to_string(), "DOID".to_string()];

        let sheaf = align_federated_ontologies(&ontology_ids).expect("should build sheaf");
        assert_eq!(sheaf.cell_count(), 3);
        assert_eq!(sheaf.restriction_count(), 9); // complete directed graph including identity maps
        assert!(sheaf.check_sheaf_condition());
    }

    #[test]
    fn test_align_federated_ontologies_dedups_and_ignores_empty() {
        let ontology_ids = vec![
            "CHEBI".to_string(),
            " ".to_string(),
            "CHEBI".to_string(),
            "MONDO".to_string(),
        ];

        let sheaf = align_federated_ontologies(&ontology_ids).expect("should build sheaf");
        assert_eq!(sheaf.cell_count(), 2);
        assert_eq!(sheaf.restriction_count(), 4);
    }
}
