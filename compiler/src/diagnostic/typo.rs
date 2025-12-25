//! Typo detection and "did you mean" suggestions
//!
//! This module provides fuzzy string matching algorithms to detect typos
//! and suggest similar names when identifiers are not found.

use std::collections::HashMap;

/// Levenshtein edit distance between two strings
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut matrix = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m {
        matrix[i][0] = i;
    }
    for j in 0..=n {
        matrix[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            matrix[i][j] = (matrix[i - 1][j] + 1) // deletion
                .min(matrix[i][j - 1] + 1) // insertion
                .min(matrix[i - 1][j - 1] + cost); // substitution
        }
    }

    matrix[m][n]
}

/// Damerau-Levenshtein distance (allows transpositions)
pub fn damerau_levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut matrix = vec![vec![0; n + 1]; m + 1];

    for i in 0..=m {
        matrix[i][0] = i;
    }
    for j in 0..=n {
        matrix[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };

            matrix[i][j] = (matrix[i - 1][j] + 1)
                .min(matrix[i][j - 1] + 1)
                .min(matrix[i - 1][j - 1] + cost);

            // Transposition
            if i > 1
                && j > 1
                && a_chars[i - 1] == b_chars[j - 2]
                && a_chars[i - 2] == b_chars[j - 1]
            {
                matrix[i][j] = matrix[i][j].min(matrix[i - 2][j - 2] + cost);
            }
        }
    }

    matrix[m][n]
}

/// Jaro similarity (0.0 to 1.0)
pub fn jaro_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();

    let match_distance = (a_chars.len().max(b_chars.len()) / 2).saturating_sub(1);

    let mut a_matches = vec![false; a_chars.len()];
    let mut b_matches = vec![false; b_chars.len()];

    let mut matches = 0;
    let mut transpositions = 0;

    for (i, &a_char) in a_chars.iter().enumerate() {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(b_chars.len());

        for j in start..end {
            if b_matches[j] || a_char != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    let mut k = 0;
    for (i, &matched) in a_matches.iter().enumerate() {
        if !matched {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let matches = matches as f64;
    let m = a_chars.len() as f64;
    let n = b_chars.len() as f64;
    let t = (transpositions as f64) / 2.0;

    (matches / m + matches / n + (matches - t) / matches) / 3.0
}

/// Jaro-Winkler similarity (favors common prefix)
pub fn jaro_winkler_similarity(a: &str, b: &str, prefix_scale: f64) -> f64 {
    let jaro = jaro_similarity(a, b);

    // Common prefix length (max 4)
    let prefix_len = a
        .chars()
        .zip(b.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    jaro + (prefix_len as f64 * prefix_scale * (1.0 - jaro))
}

/// A suggestion with similarity score
#[derive(Debug, Clone)]
pub struct TypoSuggestion {
    /// The suggested text
    pub text: String,
    /// Similarity score (higher is better)
    pub score: f64,
    /// Edit distance
    pub distance: usize,
}

/// Find similar strings from candidates
pub fn find_similar<'a>(
    query: &str,
    candidates: impl IntoIterator<Item = &'a str>,
    max_distance: usize,
    max_results: usize,
) -> Vec<TypoSuggestion> {
    let mut suggestions: Vec<TypoSuggestion> = candidates
        .into_iter()
        .filter_map(|candidate| {
            let distance = damerau_levenshtein_distance(query, candidate);
            if distance <= max_distance {
                let score = jaro_winkler_similarity(query, candidate, 0.1);
                Some(TypoSuggestion {
                    text: candidate.to_string(),
                    score,
                    distance,
                })
            } else {
                None
            }
        })
        .collect();

    // Sort by score descending, then distance ascending
    suggestions.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.distance.cmp(&b.distance))
    });

    suggestions.truncate(max_results);
    suggestions
}

/// Typo detector for identifiers
pub struct TypoDetector {
    /// Known names by category
    categories: HashMap<String, Vec<String>>,

    /// Maximum edit distance for suggestions
    max_distance: usize,

    /// Maximum suggestions per query
    max_suggestions: usize,
}

impl TypoDetector {
    /// Create a new typo detector
    pub fn new() -> Self {
        TypoDetector {
            categories: HashMap::new(),
            max_distance: 3,
            max_suggestions: 5,
        }
    }

    /// Set maximum edit distance
    pub fn with_max_distance(mut self, distance: usize) -> Self {
        self.max_distance = distance;
        self
    }

    /// Set maximum number of suggestions
    pub fn with_max_suggestions(mut self, count: usize) -> Self {
        self.max_suggestions = count;
        self
    }

    /// Add names to a category
    pub fn add_names(&mut self, category: &str, names: impl IntoIterator<Item = String>) {
        self.categories
            .entry(category.to_string())
            .or_default()
            .extend(names);
    }

    /// Add a single name to a category
    pub fn add_name(&mut self, category: &str, name: String) {
        self.categories
            .entry(category.to_string())
            .or_default()
            .push(name);
    }

    /// Find suggestions for a name in a specific category
    pub fn suggest(&self, query: &str, category: &str) -> Vec<TypoSuggestion> {
        if let Some(names) = self.categories.get(category) {
            find_similar(
                query,
                names.iter().map(|s| s.as_str()),
                self.max_distance,
                self.max_suggestions,
            )
        } else {
            Vec::new()
        }
    }

    /// Find suggestions across all categories
    pub fn suggest_all(&self, query: &str) -> Vec<(String, TypoSuggestion)> {
        let mut all_suggestions = Vec::new();

        for (category, names) in &self.categories {
            let suggestions = find_similar(
                query,
                names.iter().map(|s| s.as_str()),
                self.max_distance,
                self.max_suggestions,
            );

            for s in suggestions {
                all_suggestions.push((category.clone(), s));
            }
        }

        // Sort all suggestions by score
        all_suggestions.sort_by(|(_, a), (_, b)| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        all_suggestions.truncate(self.max_suggestions);
        all_suggestions
    }

    /// Check if a name exists in any category
    pub fn contains(&self, name: &str) -> bool {
        self.categories
            .values()
            .any(|names| names.contains(&name.to_string()))
    }

    /// Get category for a name
    pub fn category_of(&self, name: &str) -> Option<&str> {
        for (category, names) in &self.categories {
            if names.iter().any(|n| n == name) {
                return Some(category);
            }
        }
        None
    }
}

impl Default for TypoDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Context-aware suggestion builder
pub struct SuggestionBuilder {
    /// Typo detector with known names
    detector: TypoDetector,

    /// Sounio language keywords
    keywords: Vec<&'static str>,

    /// Built-in types
    builtin_types: Vec<&'static str>,

    /// Built-in functions
    builtin_fns: Vec<&'static str>,
}

impl SuggestionBuilder {
    /// Create a new suggestion builder
    pub fn new() -> Self {
        SuggestionBuilder {
            detector: TypoDetector::new(),
            keywords: vec![
                "fn", "let", "var", "const", "struct", "enum", "type", "trait", "impl", "if",
                "else", "match", "while", "for", "loop", "return", "break", "continue", "with",
                "where", "as", "in", "use", "mod", "pub", "mut", "ref", "linear", "affine",
                "kernel", "effect", "handler", "handle", "perform", "resume", "async", "await",
                "spawn", "extern", "unsafe", "static", "module", "import", "export",
            ],
            builtin_types: vec![
                "int", "i8", "i16", "i32", "i64", "i128", "uint", "u8", "u16", "u32", "u64",
                "u128", "f32", "f64", "bool", "char", "string", "unit", "never", "Option",
                "Result", "Vec", "HashMap", "HashSet", "Box", "Rc", "Arc",
            ],
            builtin_fns: vec![
                "print",
                "println",
                "assert",
                "panic",
                "todo",
                "unreachable",
                "Some",
                "None",
                "Ok",
                "Err",
                "len",
                "size",
                "clone",
                "copy",
                "drop",
                "move",
            ],
        }
    }

    /// Add scope names (variables, functions in current scope)
    pub fn with_scope_names(mut self, names: Vec<String>) -> Self {
        self.detector.add_names("scope", names);
        self
    }

    /// Add type names from current context
    pub fn with_type_names(mut self, names: Vec<String>) -> Self {
        self.detector.add_names("types", names);
        self
    }

    /// Add function names from current context
    pub fn with_function_names(mut self, names: Vec<String>) -> Self {
        self.detector.add_names("functions", names);
        self
    }

    /// Build "did you mean" message for a variable
    pub fn did_you_mean_variable(&self, query: &str) -> Option<String> {
        let mut candidates: Vec<&str> = Vec::new();

        // Add scope names
        if let Some(names) = self.detector.categories.get("scope") {
            candidates.extend(names.iter().map(|s| s.as_str()));
        }

        // Add builtin functions (they might have meant a function)
        candidates.extend(self.builtin_fns.iter().copied());

        let suggestions = find_similar(query, candidates, 3, 3);
        format_did_you_mean(&suggestions)
    }

    /// Build "did you mean" message for a type
    pub fn did_you_mean_type(&self, query: &str) -> Option<String> {
        let mut candidates: Vec<&str> = self.builtin_types.to_vec();

        // Add user-defined types
        if let Some(names) = self.detector.categories.get("types") {
            candidates.extend(names.iter().map(|s| s.as_str()));
        }

        let suggestions = find_similar(query, candidates, 3, 3);
        format_did_you_mean(&suggestions)
    }

    /// Build "did you mean" message for a function
    pub fn did_you_mean_function(&self, query: &str) -> Option<String> {
        let mut candidates: Vec<&str> = self.builtin_fns.to_vec();

        // Add user-defined functions
        if let Some(names) = self.detector.categories.get("functions") {
            candidates.extend(names.iter().map(|s| s.as_str()));
        }

        // Add scope names (might be a local function)
        if let Some(names) = self.detector.categories.get("scope") {
            candidates.extend(names.iter().map(|s| s.as_str()));
        }

        let suggestions = find_similar(query, candidates, 3, 3);
        format_did_you_mean(&suggestions)
    }

    /// Build "did you mean" message for a keyword
    pub fn did_you_mean_keyword(&self, query: &str) -> Option<String> {
        let suggestions = find_similar(query, self.keywords.iter().copied(), 2, 3);
        format_did_you_mean(&suggestions)
    }

    /// Build "did you mean" message for a field
    pub fn did_you_mean_field(&self, query: &str, field_names: &[&str]) -> Option<String> {
        let suggestions = find_similar(query, field_names.iter().copied(), 3, 3);
        format_did_you_mean(&suggestions)
    }

    /// Build "did you mean" message for a method
    pub fn did_you_mean_method(&self, query: &str, method_names: &[&str]) -> Option<String> {
        let suggestions = find_similar(query, method_names.iter().copied(), 3, 3);
        format_did_you_mean(&suggestions)
    }
}

impl Default for SuggestionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Format suggestions into a "did you mean" message
fn format_did_you_mean(suggestions: &[TypoSuggestion]) -> Option<String> {
    if suggestions.is_empty() {
        None
    } else if suggestions.len() == 1 {
        Some(format!("did you mean `{}`?", suggestions[0].text))
    } else {
        let names: Vec<_> = suggestions
            .iter()
            .take(3)
            .map(|s| format!("`{}`", s.text))
            .collect();
        Some(format!("did you mean one of: {}?", names.join(", ")))
    }
}

/// Check if query looks like a common typo pattern
pub fn is_common_typo(query: &str, target: &str) -> bool {
    if query.len() != target.len() && (query.len() as i32 - target.len() as i32).abs() > 1 {
        return false;
    }

    let distance = damerau_levenshtein_distance(query, target);

    // Single character errors are common typos
    distance == 1
}

/// Detect if two strings differ only by case
pub fn differs_only_by_case(a: &str, b: &str) -> bool {
    a.to_lowercase() == b.to_lowercase() && a != b
}

/// Suggest case correction if applicable
pub fn suggest_case_correction(query: &str, candidates: &[&str]) -> Option<String> {
    for candidate in candidates {
        if differs_only_by_case(query, candidate) {
            return Some(format!(
                "D is case-sensitive, did you mean `{}`?",
                candidate
            ));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "ab"), 1);
    }

    #[test]
    fn test_damerau_levenshtein() {
        // Transposition should cost 1, not 2
        assert_eq!(damerau_levenshtein_distance("ab", "ba"), 1);
        // "ca" -> "abc" requires insert 'b' + transpose or other operations
        assert_eq!(damerau_levenshtein_distance("ca", "abc"), 3);
    }

    #[test]
    fn test_find_similar() {
        let names = vec!["println", "print", "panic", "parse"];
        let suggestions = find_similar("prnt", names.iter().map(|s| *s), 2, 3);
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].text, "print");
    }

    #[test]
    fn test_jaro_similarity() {
        let sim = jaro_similarity("MARTHA", "MARHTA");
        assert!(sim > 0.9);

        let sim = jaro_similarity("abc", "abc");
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_typo_detector() {
        let mut detector = TypoDetector::new();
        detector.add_names(
            "variables",
            vec!["counter".into(), "count".into(), "result".into()],
        );

        let suggestions = detector.suggest("couner", "variables");
        assert!(!suggestions.is_empty());
        assert_eq!(suggestions[0].text, "counter");
    }

    #[test]
    fn test_suggestion_builder() {
        let builder = SuggestionBuilder::new()
            .with_scope_names(vec!["myVariable".into(), "myFunction".into()]);

        let msg = builder.did_you_mean_variable("myVarible");
        assert!(msg.is_some());
        assert!(msg.unwrap().contains("myVariable"));
    }

    #[test]
    fn test_case_sensitivity() {
        assert!(differs_only_by_case("Print", "print"));
        assert!(!differs_only_by_case("print", "print"));
        assert!(!differs_only_by_case("print", "println"));
    }

    #[test]
    fn test_common_typo() {
        assert!(is_common_typo("teh", "the"));
        assert!(is_common_typo("pritn", "print"));
        assert!(!is_common_typo("xyz", "print"));
    }
}
