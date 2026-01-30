//! Suggestion Engine for Type Errors
//!
//! "Did you mean?" suggestions based on semantic distance,
//! considering the current ontology context.

use std::collections::HashMap;
use std::sync::Arc;

use crate::hir::HirType;
use crate::ontology::distance::SemanticDistanceIndex;
use crate::ontology::loader::IRI;

/// A scored type suggestion
#[derive(Debug, Clone)]
pub struct ScoredSuggestion {
    /// The suggested type
    pub suggested_type: HirType,
    /// Score (0.0 to 1.0, higher is better)
    pub score: f32,
    /// Distance from the original type
    pub distance: f32,
    /// Human-readable reason
    pub reason: String,
    /// Category of suggestion
    pub category: SuggestionCategory,
}

/// Category of suggestion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuggestionCategory {
    /// Same ontology, closely related term
    SameOntologyRelated,
    /// Same ontology, sibling term
    SameOntologySibling,
    /// Different ontology, equivalent term
    CrossOntologyEquivalent,
    /// Different ontology, similar term
    CrossOntologySimilar,
    /// Subtype of expected type
    Subtype,
    /// Supertype of expected type
    Supertype,
    /// Lexically similar name
    LexicallySimilar,
}

impl SuggestionCategory {
    /// Get human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            SuggestionCategory::SameOntologyRelated => "related type in same ontology",
            SuggestionCategory::SameOntologySibling => "sibling type in same ontology",
            SuggestionCategory::CrossOntologyEquivalent => "equivalent type in different ontology",
            SuggestionCategory::CrossOntologySimilar => "similar type in different ontology",
            SuggestionCategory::Subtype => "more specific type",
            SuggestionCategory::Supertype => "more general type",
            SuggestionCategory::LexicallySimilar => "similarly named type",
        }
    }
}

/// Configuration for suggestion generation
#[derive(Debug, Clone)]
pub struct SuggestionConfig {
    /// Maximum number of suggestions to return
    pub max_suggestions: usize,
    /// Maximum distance to consider for suggestions
    pub max_distance: f32,
    /// Minimum score to include
    pub min_score: f32,
    /// Whether to include cross-ontology suggestions
    pub include_cross_ontology: bool,
    /// Whether to include subtype suggestions
    pub include_subtypes: bool,
    /// Whether to include supertype suggestions
    pub include_supertypes: bool,
}

impl Default for SuggestionConfig {
    fn default() -> Self {
        Self {
            max_suggestions: 5,
            max_distance: 0.50,
            min_score: 0.30,
            include_cross_ontology: true,
            include_subtypes: true,
            include_supertypes: true,
        }
    }
}

/// Engine for generating type suggestions
pub struct SuggestionEngine {
    /// Distance index for finding similar types
    distance_index: Arc<SemanticDistanceIndex>,
    /// Configuration
    config: SuggestionConfig,
    /// Known type aliases (expanded -> canonical)
    type_aliases: HashMap<String, String>,
    /// Common typos (wrong -> correct)
    common_typos: HashMap<String, Vec<String>>,
}

impl SuggestionEngine {
    pub fn new(distance_index: Arc<SemanticDistanceIndex>) -> Self {
        Self {
            distance_index,
            config: SuggestionConfig::default(),
            type_aliases: Self::default_aliases(),
            common_typos: Self::default_typos(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(
        distance_index: Arc<SemanticDistanceIndex>,
        config: SuggestionConfig,
    ) -> Self {
        Self {
            distance_index,
            config,
            type_aliases: Self::default_aliases(),
            common_typos: Self::default_typos(),
        }
    }

    /// Default type aliases
    fn default_aliases() -> HashMap<String, String> {
        let mut aliases = HashMap::new();
        // Common PBPK aliases
        aliases.insert("Conc".to_string(), "Concentration".to_string());
        aliases.insert("Vol".to_string(), "Volume".to_string());
        aliases.insert("Amt".to_string(), "Amount".to_string());
        aliases.insert("Clearance".to_string(), "CL".to_string());
        aliases
    }

    /// Common typos
    fn default_typos() -> HashMap<String, Vec<String>> {
        let mut typos = HashMap::new();
        typos.insert(
            "Concentraiton".to_string(),
            vec!["Concentration".to_string()],
        );
        typos.insert("Volum".to_string(), vec!["Volume".to_string()]);
        typos.insert("Absorbtion".to_string(), vec!["Absorption".to_string()]);
        typos.insert("Distributon".to_string(), vec!["Distribution".to_string()]);
        typos.insert("Metabolsim".to_string(), vec!["Metabolism".to_string()]);
        typos
    }

    /// Generate suggestions for a type mismatch
    pub fn suggest(
        &self,
        expected: &HirType,
        found: &HirType,
        context: &SuggestionContext,
    ) -> Vec<ScoredSuggestion> {
        let mut suggestions = Vec::new();

        // For ontological types, use semantic distance
        if let (
            HirType::Ontology {
                namespace: ns_e,
                term: t_e,
            },
            HirType::Ontology {
                namespace: ns_f,
                term: t_f,
            },
        ) = (expected, found)
        {
            suggestions.extend(self.suggest_ontological(ns_e, t_e, ns_f, t_f, context));
        }

        // Check for typos
        suggestions.extend(self.suggest_from_typos(expected, found));

        // Check for aliases
        suggestions.extend(self.suggest_from_aliases(expected, found));

        // Sort by score (descending) and take top N
        suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        suggestions.truncate(self.config.max_suggestions);

        // Filter by minimum score
        suggestions
            .into_iter()
            .filter(|s| s.score >= self.config.min_score)
            .collect()
    }

    /// Generate suggestions for ontological types using semantic distance
    fn suggest_ontological(
        &self,
        ns_expected: &str,
        term_expected: &str,
        ns_found: &str,
        term_found: &str,
        _context: &SuggestionContext,
    ) -> Vec<ScoredSuggestion> {
        let mut suggestions = Vec::new();

        let iri_expected = IRI::from_curie(ns_expected, term_expected);
        let iri_found = IRI::from_curie(ns_found, term_found);

        // Calculate the actual distance between found and expected
        let semantic_distance = self.distance_index.distance(&iri_found, &iri_expected);
        let distance = semantic_distance.conceptual as f32;

        // If they're somewhat close, suggest the expected type directly
        if distance < self.config.max_distance {
            let score = 1.0 - distance;
            let category = if ns_expected == ns_found {
                SuggestionCategory::SameOntologyRelated
            } else {
                SuggestionCategory::CrossOntologySimilar
            };

            suggestions.push(ScoredSuggestion {
                suggested_type: HirType::Ontology {
                    namespace: ns_expected.to_string(),
                    term: term_expected.to_string(),
                },
                score,
                distance,
                reason: format!(
                    "{}: `{}:{}`",
                    category.description(),
                    ns_expected,
                    term_expected
                ),
                category,
            });
        }

        suggestions
    }

    /// Suggest based on common typos
    fn suggest_from_typos(&self, expected: &HirType, found: &HirType) -> Vec<ScoredSuggestion> {
        let mut suggestions = Vec::new();

        if let HirType::Ontology {
            term: found_term, ..
        } = found
            && let Some(corrections) = self.common_typos.get(found_term)
        {
            for correction in corrections {
                // Check if correction matches expected
                if let HirType::Ontology {
                    namespace: ns_e,
                    term: t_e,
                } = expected
                    && t_e == correction
                {
                    suggestions.push(ScoredSuggestion {
                        suggested_type: HirType::Ontology {
                            namespace: ns_e.clone(),
                            term: correction.clone(),
                        },
                        score: 0.95, // High score for typo fix
                        distance: 0.02,
                        reason: format!("did you mean `{}`? (typo)", correction),
                        category: SuggestionCategory::LexicallySimilar,
                    });
                }
            }
        }

        suggestions
    }

    /// Suggest based on aliases
    fn suggest_from_aliases(&self, expected: &HirType, found: &HirType) -> Vec<ScoredSuggestion> {
        let mut suggestions = Vec::new();

        if let (
            HirType::Ontology {
                namespace: ns_e,
                term: t_e,
            },
            HirType::Ontology { term: t_f, .. },
        ) = (expected, found)
        {
            // Check if found is an alias of expected
            if let Some(canonical) = self.type_aliases.get(t_f)
                && canonical == t_e
            {
                suggestions.push(ScoredSuggestion {
                    suggested_type: expected.clone(),
                    score: 0.90,
                    distance: 0.05,
                    reason: format!("`{}` is an alias for `{}`", t_f, t_e),
                    category: SuggestionCategory::LexicallySimilar,
                });
            }

            // Check reverse: if expected is an alias
            if let Some(canonical) = self.type_aliases.get(t_e)
                && canonical == t_f
            {
                suggestions.push(ScoredSuggestion {
                    suggested_type: found.clone(),
                    score: 0.90,
                    distance: 0.05,
                    reason: format!("`{}` is an alias for `{}`", t_e, t_f),
                    category: SuggestionCategory::LexicallySimilar,
                });
            }
        }

        suggestions
    }

    /// Add a type alias
    pub fn add_alias(&mut self, alias: &str, canonical: &str) {
        self.type_aliases
            .insert(alias.to_string(), canonical.to_string());
    }

    /// Add a common typo
    pub fn add_typo(&mut self, typo: &str, corrections: Vec<String>) {
        self.common_typos.insert(typo.to_string(), corrections);
    }
}

/// Context for suggestion generation
#[derive(Debug, Clone, Default)]
pub struct SuggestionContext {
    /// Current module path
    pub module_path: String,
    /// Current function (if any)
    pub function_name: Option<String>,
    /// Types in scope
    pub types_in_scope: Vec<HirType>,
    /// Recently used types
    pub recent_types: Vec<HirType>,
    /// Imported namespaces
    pub imported_namespaces: Vec<String>,
}

impl SuggestionContext {
    pub fn new(module_path: &str) -> Self {
        Self {
            module_path: module_path.to_string(),
            ..Default::default()
        }
    }

    pub fn with_function(mut self, name: &str) -> Self {
        self.function_name = Some(name.to_string());
        self
    }

    pub fn add_type_in_scope(&mut self, ty: HirType) {
        self.types_in_scope.push(ty);
    }

    pub fn add_recent_type(&mut self, ty: HirType) {
        // Keep limited history
        if self.recent_types.len() >= 10 {
            self.recent_types.remove(0);
        }
        self.recent_types.push(ty);
    }

    pub fn add_imported_namespace(&mut self, ns: &str) {
        self.imported_namespaces.push(ns.to_string());
    }
}

/// Calculate Levenshtein edit distance between two strings
pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
    let len1 = s1.chars().count();
    let len2 = s2.chars().count();

    if len1 == 0 {
        return len2;
    }
    if len2 == 0 {
        return len1;
    }

    let s1_chars: Vec<char> = s1.chars().collect();
    let s2_chars: Vec<char> = s2.chars().collect();

    let mut matrix = vec![vec![0usize; len2 + 1]; len1 + 1];

    for i in 0..=len1 {
        matrix[i][0] = i;
    }
    for j in 0..=len2 {
        matrix[0][j] = j;
    }

    for i in 1..=len1 {
        for j in 1..=len2 {
            let cost = if s1_chars[i - 1] == s2_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);
        }
    }

    matrix[len1][len2]
}

/// Normalize Levenshtein to 0-1 range
pub fn normalized_levenshtein(s1: &str, s2: &str) -> f32 {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return 0.0;
    }
    levenshtein_distance(s1, s2) as f32 / max_len as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
    }

    #[test]
    fn test_normalized_levenshtein() {
        let dist = normalized_levenshtein("Concentration", "Concentraiton");
        assert!(dist < 0.2); // Close strings

        let dist = normalized_levenshtein("Absorption", "Metabolism");
        assert!(dist > 0.5); // Different strings
    }

    #[test]
    fn test_suggestion_category() {
        assert_eq!(
            SuggestionCategory::SameOntologyRelated.description(),
            "related type in same ontology"
        );
        assert_eq!(
            SuggestionCategory::CrossOntologyEquivalent.description(),
            "equivalent type in different ontology"
        );
    }

    #[test]
    fn test_scored_suggestion_ordering() {
        let s1 = ScoredSuggestion {
            suggested_type: HirType::I64,
            score: 0.8,
            distance: 0.2,
            reason: "test".to_string(),
            category: SuggestionCategory::Subtype,
        };
        let s2 = ScoredSuggestion {
            suggested_type: HirType::I32,
            score: 0.6,
            distance: 0.4,
            reason: "test".to_string(),
            category: SuggestionCategory::Subtype,
        };

        let mut suggestions = vec![s2.clone(), s1.clone()];
        suggestions.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        assert_eq!(suggestions[0].score, 0.8);
        assert_eq!(suggestions[1].score, 0.6);
    }
}
