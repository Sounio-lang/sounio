//! Ontology Validator for LLM-Generated Content
//!
//! Validates generated ontology fragments for:
//! - Logical consistency
//! - BFO alignment
//! - Structural issues
//! - Naming conventions
//! - Documentation completeness

use super::{GeneratedAxiom, GeneratedOntologyFragment};
use std::collections::{HashMap, HashSet};

/// Configuration for validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check logical consistency
    pub check_consistency: bool,
    /// Check BFO alignment
    pub check_bfo_alignment: bool,
    /// Check naming conventions
    pub check_naming: bool,
    /// Check documentation
    pub check_documentation: bool,
    /// Minimum confidence for warnings
    pub min_confidence_warning: f64,
    /// Minimum confidence for errors
    pub min_confidence_error: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_consistency: true,
            check_bfo_alignment: true,
            check_naming: true,
            check_documentation: true,
            min_confidence_warning: 0.6,
            min_confidence_error: 0.4,
        }
    }
}

/// Ontology validator
pub struct OntologyValidator {
    config: ValidationConfig,
    /// Known BFO classes
    bfo_classes: HashSet<String>,
    /// Known RO relations
    ro_relations: HashSet<String>,
}

impl OntologyValidator {
    /// Create a new validator with default config
    pub fn new() -> Self {
        Self::with_config(ValidationConfig::default())
    }

    /// Create with custom config
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            bfo_classes: Self::init_bfo_classes(),
            ro_relations: Self::init_ro_relations(),
        }
    }

    /// Initialize known BFO classes
    fn init_bfo_classes() -> HashSet<String> {
        let classes = [
            "BFO:0000001", // entity
            "BFO:0000002", // continuant
            "BFO:0000003", // occurrent
            "BFO:0000004", // independent continuant
            "BFO:0000015", // process
            "BFO:0000016", // disposition
            "BFO:0000017", // realizable entity
            "BFO:0000019", // quality
            "BFO:0000020", // specifically dependent continuant
            "BFO:0000023", // role
            "BFO:0000024", // fiat object part
            "BFO:0000027", // object aggregate
            "BFO:0000029", // site
            "BFO:0000030", // object
            "BFO:0000031", // generically dependent continuant
            "BFO:0000034", // function
            "BFO:0000035", // process boundary
            "BFO:0000038", // one-dimensional temporal region
            "BFO:0000040", // material entity
            "BFO:0000140", // continuant fiat boundary
            "BFO:0000141", // immaterial entity
            "BFO:0000142", // one-dimensional continuant fiat boundary
            "BFO:0000144", // process profile
            "BFO:0000145", // relational quality
            "BFO:0000146", // two-dimensional continuant fiat boundary
            "BFO:0000147", // zero-dimensional continuant fiat boundary
            "BFO:0000148", // zero-dimensional temporal region
            "BFO:0000182", // history
            "BFO:0000202", // temporal interval
            "BFO:0000203", // temporal instant
        ];
        classes.iter().map(|s| s.to_string()).collect()
    }

    /// Initialize known RO relations
    fn init_ro_relations() -> HashSet<String> {
        let relations = [
            "RO:0000052",  // inheres in
            "RO:0000053",  // bearer of
            "RO:0000056",  // participates in
            "RO:0000057",  // has participant
            "RO:0000058",  // is concretized as
            "RO:0000059",  // concretizes
            "RO:0000080",  // quality of
            "RO:0000081",  // role of
            "RO:0000086",  // has quality
            "RO:0000087",  // has role
            "RO:0000091",  // has disposition
            "RO:0002131",  // overlaps
            "RO:0002162",  // in taxon
            "RO:0002350",  // member of
            "BFO:0000050", // part of
            "BFO:0000051", // has part
            "BFO:0000066", // occurs in
            "BFO:0000067", // contains process
        ];
        relations.iter().map(|s| s.to_string()).collect()
    }

    /// Validate an ontology fragment
    pub fn validate(&self, fragment: &GeneratedOntologyFragment) -> ValidationResult {
        let mut issues = Vec::new();

        // Consistency checks
        if self.config.check_consistency {
            issues.extend(self.check_consistency(fragment));
        }

        // BFO alignment
        if self.config.check_bfo_alignment {
            issues.extend(self.check_bfo_alignment(fragment));
        }

        // Naming conventions
        if self.config.check_naming {
            issues.extend(self.check_naming(fragment));
        }

        // Documentation
        if self.config.check_documentation {
            issues.extend(self.check_documentation(fragment));
        }

        // Confidence checks
        issues.extend(self.check_confidence(fragment));

        ValidationResult::new(issues)
    }

    /// Check logical consistency
    fn check_consistency(&self, fragment: &GeneratedOntologyFragment) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        // Build class hierarchy graph
        let mut subclass_of: HashMap<&str, Vec<&str>> = HashMap::new();
        for axiom in &fragment.axioms {
            if let GeneratedAxiom::SubClassOf {
                subclass,
                superclass,
                ..
            } = axiom
            {
                subclass_of
                    .entry(subclass.as_str())
                    .or_default()
                    .push(superclass.as_str());
            }
        }

        // Check for cycles
        for class in &fragment.classes {
            if self.has_cycle(&class.name, &subclass_of, &mut HashSet::new()) {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: "consistency".to_string(),
                    element: class.name.clone(),
                    message: "Cycle detected in class hierarchy".to_string(),
                    suggestion: Some("Remove circular subclass relationships".to_string()),
                });
            }
        }

        // Check for disjoint violations
        let mut disjoint_pairs: HashSet<(&str, &str)> = HashSet::new();
        for axiom in &fragment.axioms {
            if let GeneratedAxiom::DisjointWith { class1, class2, .. } = axiom {
                disjoint_pairs.insert((class1.as_str(), class2.as_str()));
                disjoint_pairs.insert((class2.as_str(), class1.as_str()));
            }
        }

        // Check if any class is subclass of two disjoint classes
        for (class, parents) in &subclass_of {
            for i in 0..parents.len() {
                for j in (i + 1)..parents.len() {
                    if disjoint_pairs.contains(&(parents[i], parents[j])) {
                        issues.push(ValidationIssue {
                            severity: ValidationSeverity::Error,
                            category: "consistency".to_string(),
                            element: class.to_string(),
                            message: format!(
                                "Class is subclass of disjoint classes '{}' and '{}'",
                                parents[i], parents[j]
                            ),
                            suggestion: Some("Review class hierarchy".to_string()),
                        });
                    }
                }
            }
        }

        // Check for orphan classes (no parent except BFO)
        let classes_with_parents: HashSet<_> = subclass_of.keys().copied().collect();
        for class in &fragment.classes {
            if !classes_with_parents.contains(class.name.as_str()) && !class.bfo_parent.is_empty() {
                // Has BFO parent, which is fine
            } else if class.bfo_parent.is_empty() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: "structure".to_string(),
                    element: class.name.clone(),
                    message: "Class has no BFO parent".to_string(),
                    suggestion: Some("Assign an appropriate BFO category".to_string()),
                });
            }
        }

        issues
    }

    /// Check for cycles in class hierarchy
    fn has_cycle(
        &self,
        class: &str,
        subclass_of: &HashMap<&str, Vec<&str>>,
        visited: &mut HashSet<String>,
    ) -> bool {
        if visited.contains(class) {
            return true;
        }

        visited.insert(class.to_string());

        if let Some(parents) = subclass_of.get(class) {
            for parent in parents {
                if self.has_cycle(parent, subclass_of, visited) {
                    return true;
                }
            }
        }

        visited.remove(class);
        false
    }

    /// Check BFO alignment
    fn check_bfo_alignment(&self, fragment: &GeneratedOntologyFragment) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        for class in &fragment.classes {
            // Check if BFO parent is valid
            if !class.bfo_parent.is_empty() && !self.bfo_classes.contains(&class.bfo_parent) {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: "bfo_alignment".to_string(),
                    element: class.name.clone(),
                    message: format!("Unknown BFO class: {}", class.bfo_parent),
                    suggestion: Some("Use a valid BFO class identifier".to_string()),
                });
            }
        }

        // Check property parents
        for prop in &fragment.properties {
            if let Some(ref ro_parent) = prop.ro_parent
                && !self.ro_relations.contains(ro_parent)
            {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: "bfo_alignment".to_string(),
                    element: prop.name.clone(),
                    message: format!("Unknown RO relation: {}", ro_parent),
                    suggestion: Some("Use a valid RO relation identifier".to_string()),
                });
            }
        }

        issues
    }

    /// Check naming conventions
    fn check_naming(&self, fragment: &GeneratedOntologyFragment) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        for class in &fragment.classes {
            // Classes should be CamelCase or contain spaces (labels)
            if class.name.contains('_') && !class.name.contains(' ') {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Suggestion,
                    category: "naming".to_string(),
                    element: class.name.clone(),
                    message: "Class name uses snake_case instead of CamelCase".to_string(),
                    suggestion: Some(format!(
                        "Consider renaming to '{}'",
                        to_camel_case(&class.name)
                    )),
                });
            }

            // Check for very short names
            if class.name.len() < 3 {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: "naming".to_string(),
                    element: class.name.clone(),
                    message: "Class name is very short".to_string(),
                    suggestion: Some("Use a more descriptive name".to_string()),
                });
            }

            // Check for duplicate names (case-insensitive)
            let lower_name = class.name.to_lowercase();
            let duplicates: Vec<_> = fragment
                .classes
                .iter()
                .filter(|c| c.name.to_lowercase() == lower_name && c.name != class.name)
                .collect();
            if !duplicates.is_empty() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: "naming".to_string(),
                    element: class.name.clone(),
                    message: "Duplicate class name (case-insensitive)".to_string(),
                    suggestion: Some("Use unique class names".to_string()),
                });
            }
        }

        for prop in &fragment.properties {
            // Properties should be snake_case or camelCase
            if prop.name.contains(' ') {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Suggestion,
                    category: "naming".to_string(),
                    element: prop.name.clone(),
                    message: "Property name contains spaces".to_string(),
                    suggestion: Some(format!(
                        "Consider renaming to '{}'",
                        prop.name.replace(' ', "_").to_lowercase()
                    )),
                });
            }
        }

        issues
    }

    /// Check documentation completeness
    fn check_documentation(&self, fragment: &GeneratedOntologyFragment) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        for class in &fragment.classes {
            // Check for missing definitions
            if class.definition.is_none() {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Suggestion,
                    category: "documentation".to_string(),
                    element: class.name.clone(),
                    message: "Class has no definition".to_string(),
                    suggestion: Some("Add a formal definition".to_string()),
                });
            }

            // Check for missing labels
            if class.label.is_empty() || class.label == class.name {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Suggestion,
                    category: "documentation".to_string(),
                    element: class.name.clone(),
                    message: "Class has no human-readable label".to_string(),
                    suggestion: Some("Add a descriptive label".to_string()),
                });
            }
        }

        issues
    }

    /// Check confidence levels
    fn check_confidence(&self, fragment: &GeneratedOntologyFragment) -> Vec<ValidationIssue> {
        let mut issues = Vec::new();

        for class in &fragment.classes {
            if class.confidence < self.config.min_confidence_error {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: "confidence".to_string(),
                    element: class.name.clone(),
                    message: format!(
                        "Very low confidence ({:.2}) - likely unreliable",
                        class.confidence
                    ),
                    suggestion: Some("Manual review required".to_string()),
                });
            } else if class.confidence < self.config.min_confidence_warning {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: "confidence".to_string(),
                    element: class.name.clone(),
                    message: format!("Low confidence ({:.2})", class.confidence),
                    suggestion: Some("Consider manual verification".to_string()),
                });
            }
        }

        for axiom in &fragment.axioms {
            let (element, confidence) = match axiom {
                GeneratedAxiom::SubClassOf {
                    subclass,
                    confidence,
                    ..
                } => (format!("{}→SubClassOf", subclass), *confidence),
                GeneratedAxiom::EquivalentClass {
                    class1, confidence, ..
                } => (format!("{}→Equivalent", class1), *confidence),
                GeneratedAxiom::DisjointWith {
                    class1, confidence, ..
                } => (format!("{}→Disjoint", class1), *confidence),
                GeneratedAxiom::ObjectPropertyAssertion {
                    subject,
                    predicate,
                    confidence,
                    ..
                } => (format!("{}→{}", subject, predicate), *confidence),
                GeneratedAxiom::AnnotationAssertion { .. } => continue,
            };

            if confidence < self.config.min_confidence_error {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    category: "confidence".to_string(),
                    element,
                    message: format!("Very low axiom confidence ({:.2})", confidence),
                    suggestion: Some("Manual review required".to_string()),
                });
            } else if confidence < self.config.min_confidence_warning {
                issues.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    category: "confidence".to_string(),
                    element,
                    message: format!("Low axiom confidence ({:.2})", confidence),
                    suggestion: Some("Consider verification".to_string()),
                });
            }
        }

        issues
    }
}

impl Default for OntologyValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert snake_case to CamelCase
fn to_camel_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().chain(chars).collect(),
                None => String::new(),
            }
        })
        .collect()
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// All issues found
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Create a new result
    pub fn new(issues: Vec<ValidationIssue>) -> Self {
        Self { issues }
    }

    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| matches!(i.severity, ValidationSeverity::Error))
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, ValidationSeverity::Error))
            .count()
    }

    /// Get warning count
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, ValidationSeverity::Warning))
            .count()
    }

    /// Get suggestion count
    pub fn suggestion_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|i| matches!(i.severity, ValidationSeverity::Suggestion))
            .count()
    }

    /// Get issues by category
    pub fn by_category(&self, category: &str) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.category == category)
            .collect()
    }

    /// Get issues by severity
    pub fn by_severity(&self, severity: ValidationSeverity) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == severity)
            .collect()
    }

    /// Get overall quality score (0.0-1.0)
    pub fn quality_score(&self) -> f64 {
        if self.issues.is_empty() {
            return 1.0;
        }

        let total_penalty: f64 = self
            .issues
            .iter()
            .map(|i| match i.severity {
                ValidationSeverity::Error => 0.3,
                ValidationSeverity::Warning => 0.1,
                ValidationSeverity::Suggestion => 0.02,
            })
            .sum();

        (1.0 - total_penalty).max(0.0)
    }
}

/// A single validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity level
    pub severity: ValidationSeverity,
    /// Category (consistency, naming, documentation, etc.)
    pub category: String,
    /// Affected element
    pub element: String,
    /// Issue description
    pub message: String,
    /// Suggested fix
    pub suggestion: Option<String>,
}

/// Issue severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Errors that must be fixed
    Error,
    /// Warnings that should be reviewed
    Warning,
    /// Suggestions for improvement
    Suggestion,
}

impl std::fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationSeverity::Error => write!(f, "error"),
            ValidationSeverity::Warning => write!(f, "warning"),
            ValidationSeverity::Suggestion => write!(f, "suggestion"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::llm_gen::{GeneratedClass, LLMProvenance};

    fn make_class(name: &str, confidence: f64, bfo_parent: &str) -> GeneratedClass {
        GeneratedClass {
            name: name.to_string(),
            label: name.to_string(),
            bfo_parent: bfo_parent.to_string(),
            bfo_parent_label: "".to_string(),
            definition: None,
            reasoning: None,
            confidence,
            provenance: LLMProvenance::new("test"),
        }
    }

    #[test]
    fn test_empty_fragment_valid() {
        let validator = OntologyValidator::new();
        let fragment = GeneratedOntologyFragment::new("test");

        let result = validator.validate(&fragment);
        assert!(result.is_valid());
        assert_eq!(result.error_count(), 0);
    }

    #[test]
    fn test_low_confidence_warning() {
        let validator = OntologyValidator::new();
        let mut fragment = GeneratedOntologyFragment::new("test");
        fragment
            .classes
            .push(make_class("LowConf", 0.5, "BFO:0000040"));

        let result = validator.validate(&fragment);
        assert!(result.warning_count() > 0);
    }

    #[test]
    fn test_very_low_confidence_error() {
        let validator = OntologyValidator::new();
        let mut fragment = GeneratedOntologyFragment::new("test");
        fragment
            .classes
            .push(make_class("VeryLowConf", 0.2, "BFO:0000040"));

        let result = validator.validate(&fragment);
        assert!(!result.is_valid());
        assert!(result.error_count() > 0);
    }

    #[test]
    fn test_unknown_bfo_class() {
        let validator = OntologyValidator::new();
        let mut fragment = GeneratedOntologyFragment::new("test");
        fragment
            .classes
            .push(make_class("Test", 0.9, "BFO:INVALID"));

        let result = validator.validate(&fragment);
        let bfo_issues = result.by_category("bfo_alignment");
        assert!(!bfo_issues.is_empty());
    }

    #[test]
    fn test_snake_case_naming() {
        let validator = OntologyValidator::new();
        let mut fragment = GeneratedOntologyFragment::new("test");
        fragment
            .classes
            .push(make_class("my_class_name", 0.9, "BFO:0000040"));

        let result = validator.validate(&fragment);
        let naming_issues = result.by_category("naming");
        assert!(!naming_issues.is_empty());
    }

    #[test]
    fn test_missing_definition_suggestion() {
        let validator = OntologyValidator::new();
        let mut fragment = GeneratedOntologyFragment::new("test");
        fragment
            .classes
            .push(make_class("NoDefinition", 0.9, "BFO:0000040"));

        let result = validator.validate(&fragment);
        let doc_issues = result.by_category("documentation");
        assert!(!doc_issues.is_empty());
    }

    #[test]
    fn test_quality_score() {
        let validator = OntologyValidator::new();

        // Perfect fragment
        let mut good_fragment = GeneratedOntologyFragment::new("test");
        good_fragment.classes.push(GeneratedClass {
            name: "GoodClass".to_string(),
            label: "Good Class".to_string(),
            bfo_parent: "BFO:0000040".to_string(),
            bfo_parent_label: "material entity".to_string(),
            definition: Some("A good class definition".to_string()),
            reasoning: None,
            confidence: 0.95,
            provenance: LLMProvenance::new("test"),
        });

        let good_result = validator.validate(&good_fragment);
        assert!(good_result.quality_score() > 0.8);

        // Fragment with issues
        let mut bad_fragment = GeneratedOntologyFragment::new("test");
        bad_fragment
            .classes
            .push(make_class("x", 0.2, "BFO:INVALID"));

        let bad_result = validator.validate(&bad_fragment);
        assert!(bad_result.quality_score() < good_result.quality_score());
    }

    #[test]
    fn test_to_camel_case() {
        assert_eq!(to_camel_case("snake_case"), "SnakeCase");
        assert_eq!(to_camel_case("my_class_name"), "MyClassName");
        assert_eq!(to_camel_case("single"), "Single");
    }
}
