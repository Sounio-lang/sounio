//! Code actions provider for quick fixes and refactorings
//!
//! Provides code actions for:
//! - Quick fixes from diagnostics (typos, missing imports, etc.)
//! - Refactoring actions (extract function, extract variable, inline, etc.)
//! - Source actions (organize imports, add missing implementations)
//!
//! # Code Action Kinds
//!
//! - `quickfix` - Fixes for diagnostics
//! - `refactor` - Generic refactoring actions
//! - `refactor.extract` - Extract to function/variable/constant
//! - `refactor.inline` - Inline function/variable
//! - `refactor.rewrite` - Rewrite expressions
//! - `source` - Source-level actions
//! - `source.organizeImports` - Organize imports

use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Item};
use crate::resolve::SymbolTable;

// ============================================================================
// Levenshtein Distance for "Did you mean?" suggestions
// ============================================================================

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

/// Calculate normalized Levenshtein distance (0.0 to 1.0)
pub fn normalized_levenshtein(s1: &str, s2: &str) -> f32 {
    let max_len = s1.len().max(s2.len());
    if max_len == 0 {
        return 0.0;
    }
    levenshtein_distance(s1, s2) as f32 / max_len as f32
}

/// Find similar names in a list using Levenshtein distance
pub fn find_similar_names(target: &str, candidates: &[String], max_distance: usize) -> Vec<(String, usize)> {
    let mut matches: Vec<(String, usize)> = candidates
        .iter()
        .filter_map(|candidate| {
            let distance = levenshtein_distance(target, candidate);
            if distance <= max_distance && distance > 0 {
                Some((candidate.clone(), distance))
            } else {
                None
            }
        })
        .collect();

    // Sort by distance (closest first)
    matches.sort_by_key(|(_, d)| *d);
    matches
}

/// Calculate similarity score (0.0 to 1.0, higher is more similar)
pub fn similarity_score(s1: &str, s2: &str) -> f32 {
    1.0 - normalized_levenshtein(s1, s2)
}

// ============================================================================
// Type Compatibility Checking
// ============================================================================

/// Represents a type for compatibility checking
#[derive(Debug, Clone, PartialEq)]
pub enum TypeKind {
    /// Integer types: i8, i16, i32, i64, i128, isize
    Int { bits: u8, signed: bool },
    /// Float types: f32, f64
    Float { bits: u8 },
    /// Boolean
    Bool,
    /// String
    String,
    /// Reference (shared)
    Ref(Box<TypeKind>),
    /// Mutable reference (exclusive, uses &!)
    RefMut(Box<TypeKind>),
    /// Named type (struct, enum, etc.)
    Named(String),
    /// Array
    Array(Box<TypeKind>),
    /// Unknown/other
    Unknown,
}

impl TypeKind {
    /// Parse a type string into TypeKind
    pub fn parse(type_str: &str) -> Self {
        let s = type_str.trim();

        // Handle references (Sounio uses &! for mutable refs)
        if s.starts_with("&!") {
            return TypeKind::RefMut(Box::new(TypeKind::parse(&s[2..])));
        }
        if s.starts_with('&') {
            return TypeKind::Ref(Box::new(TypeKind::parse(&s[1..])));
        }

        // Handle arrays
        if s.starts_with('[') && s.ends_with(']') {
            let inner = &s[1..s.len()-1];
            // Handle sized arrays [T; N]
            if let Some(semi_pos) = inner.rfind(';') {
                return TypeKind::Array(Box::new(TypeKind::parse(&inner[..semi_pos].trim())));
            }
            return TypeKind::Array(Box::new(TypeKind::parse(inner)));
        }

        match s {
            "i8" => TypeKind::Int { bits: 8, signed: true },
            "i16" => TypeKind::Int { bits: 16, signed: true },
            "i32" => TypeKind::Int { bits: 32, signed: true },
            "i64" => TypeKind::Int { bits: 64, signed: true },
            "i128" => TypeKind::Int { bits: 128, signed: true },
            "isize" => TypeKind::Int { bits: 64, signed: true }, // platform-dependent
            "u8" => TypeKind::Int { bits: 8, signed: false },
            "u16" => TypeKind::Int { bits: 16, signed: false },
            "u32" => TypeKind::Int { bits: 32, signed: false },
            "u64" => TypeKind::Int { bits: 64, signed: false },
            "u128" => TypeKind::Int { bits: 128, signed: false },
            "usize" => TypeKind::Int { bits: 64, signed: false },
            "f32" => TypeKind::Float { bits: 32 },
            "f64" => TypeKind::Float { bits: 64 },
            "bool" => TypeKind::Bool,
            "String" | "str" | "string" => TypeKind::String,
            _ => TypeKind::Named(s.to_string()),
        }
    }

    /// Check if this type can be cast to another type
    pub fn can_cast_to(&self, target: &TypeKind) -> bool {
        match (self, target) {
            // Same type, no cast needed
            (a, b) if a == b => true,

            // Numeric casts
            (TypeKind::Int { .. }, TypeKind::Int { .. }) => true,
            (TypeKind::Int { .. }, TypeKind::Float { .. }) => true,
            (TypeKind::Float { .. }, TypeKind::Int { .. }) => true,
            (TypeKind::Float { bits: 32 }, TypeKind::Float { bits: 64 }) => true,
            (TypeKind::Float { bits: 64 }, TypeKind::Float { bits: 32 }) => true,

            // Bool to int
            (TypeKind::Bool, TypeKind::Int { .. }) => true,

            // Reference compatibility
            (TypeKind::Ref(inner), TypeKind::Ref(target_inner)) => inner == target_inner,
            (TypeKind::RefMut(inner), TypeKind::Ref(target_inner)) => inner == target_inner,

            _ => false,
        }
    }

    /// Get the cast expression to convert this type to target
    pub fn cast_expr(&self, expr: &str, target: &TypeKind) -> Option<String> {
        if self == target {
            return None;
        }

        match (self, target) {
            // Numeric casts use `as`
            (TypeKind::Int { .. }, TypeKind::Int { .. }) |
            (TypeKind::Int { .. }, TypeKind::Float { .. }) |
            (TypeKind::Float { .. }, TypeKind::Int { .. }) |
            (TypeKind::Float { .. }, TypeKind::Float { .. }) |
            (TypeKind::Bool, TypeKind::Int { .. }) => {
                Some(format!("{} as {}", expr, target.type_name()))
            }

            // Reference to ref-mut needs special handling
            (TypeKind::Ref(_), TypeKind::RefMut(_)) => None, // Can't go from & to &!

            // Deref and re-ref
            (TypeKind::RefMut(inner), TypeKind::Ref(target_inner)) if inner == target_inner => {
                Some(format!("&{}", expr.trim_start_matches("&!")))
            }

            _ => None,
        }
    }

    /// Get type name as string
    pub fn type_name(&self) -> String {
        match self {
            TypeKind::Int { bits: 8, signed: true } => "i8".to_string(),
            TypeKind::Int { bits: 16, signed: true } => "i16".to_string(),
            TypeKind::Int { bits: 32, signed: true } => "i32".to_string(),
            TypeKind::Int { bits: 64, signed: true } => "i64".to_string(),
            TypeKind::Int { bits: 128, signed: true } => "i128".to_string(),
            TypeKind::Int { bits: 8, signed: false } => "u8".to_string(),
            TypeKind::Int { bits: 16, signed: false } => "u16".to_string(),
            TypeKind::Int { bits: 32, signed: false } => "u32".to_string(),
            TypeKind::Int { bits: 64, signed: false } => "u64".to_string(),
            TypeKind::Int { bits: 128, signed: false } => "u128".to_string(),
            TypeKind::Int { bits: _, signed: true } => "isize".to_string(),
            TypeKind::Int { bits: _, signed: false } => "usize".to_string(),
            TypeKind::Float { bits: 32 } => "f32".to_string(),
            TypeKind::Float { bits: 64 } => "f64".to_string(),
            TypeKind::Float { bits: _ } => "f64".to_string(), // Default to f64 for other float sizes
            TypeKind::Bool => "bool".to_string(),
            TypeKind::String => "String".to_string(),
            TypeKind::Ref(inner) => format!("&{}", inner.type_name()),
            TypeKind::RefMut(inner) => format!("&!{}", inner.type_name()),
            TypeKind::Named(name) => name.clone(),
            TypeKind::Array(inner) => format!("[{}]", inner.type_name()),
            TypeKind::Unknown => "_".to_string(),
        }
    }
}

// ============================================================================
// Effect Analysis
// ============================================================================

/// Built-in effects in Sounio
pub const BUILTIN_EFFECTS: &[&str] = &[
    "IO",     // File, network, console I/O
    "Mut",    // Mutable state
    "Alloc",  // Heap allocation
    "Panic",  // Recoverable failure
    "Async",  // Asynchronous operations
    "GPU",    // GPU kernel launch, device memory
    "Prob",   // Probabilistic computation
    "Div",    // Potential divergence
];

/// Functions and their associated effects
pub fn get_function_effects(fn_name: &str) -> Option<&'static [&'static str]> {
    match fn_name {
        // IO effects
        "print" | "println" | "dbg" | "read_line" => Some(&["IO"]),
        "read_file" | "write_file" | "open_file" => Some(&["IO"]),

        // Panic effects
        "panic" | "assert" | "assert_eq" | "unwrap" | "expect" => Some(&["Panic"]),

        // Prob effects
        "sample" | "observe" | "infer" => Some(&["Prob"]),

        // GPU effects
        "gpu_launch" | "gpu_sync" | "device_malloc" => Some(&["GPU"]),

        // Alloc effects
        "Vec::new" | "Box::new" | "String::new" => Some(&["Alloc"]),

        _ => None,
    }
}

// ============================================================================
// Code Action Provider
// ============================================================================

/// Provider for code actions
pub struct CodeActionProvider {
    /// Enable quick fixes
    pub enable_quick_fixes: bool,
    /// Enable refactoring actions
    pub enable_refactorings: bool,
    /// Enable source actions
    pub enable_source_actions: bool,
}

impl CodeActionProvider {
    /// Create a new code action provider
    pub fn new() -> Self {
        Self {
            enable_quick_fixes: true,
            enable_refactorings: true,
            enable_source_actions: true,
        }
    }

    /// Generate code actions for a range
    pub fn code_actions(
        &self,
        uri: &Url,
        range: Range,
        diagnostics: &[Diagnostic],
        source: &str,
        ast: Option<&Ast>,
        symbols: Option<&SymbolTable>,
    ) -> Vec<CodeActionOrCommand> {
        let mut actions = Vec::new();

        // Quick fixes from diagnostics
        if self.enable_quick_fixes {
            for diag in diagnostics {
                self.add_diagnostic_fixes(uri, diag, source, ast, symbols, &mut actions);
            }
        }

        // Refactoring actions based on selection
        if self.enable_refactorings {
            self.add_refactoring_actions(uri, range, source, ast, symbols, &mut actions);
        }

        // Source actions
        if self.enable_source_actions {
            self.add_source_actions(uri, source, ast, &mut actions);
        }

        actions
    }

    /// Add quick fixes for a diagnostic
    fn add_diagnostic_fixes(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        ast: Option<&Ast>,
        symbols: Option<&SymbolTable>,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Extract error code
        let code = match &diagnostic.code {
            Some(NumberOrString::String(s)) => s.as_str(),
            Some(NumberOrString::Number(n)) => {
                match n {
                    1 => "E0001",
                    2 => "E0002",
                    3 => "E0003",
                    4 => "E0004",
                    5 => "E0005",
                    6 => "E0006",
                    _ => return,
                }
            }
            None => {
                // Try to infer from message content
                let msg = diagnostic.message.to_lowercase();
                if msg.contains("type mismatch") || msg.contains("expected") && msg.contains("found") {
                    "E0001"
                } else if msg.contains("undefined") || msg.contains("not found") || msg.contains("cannot find") {
                    "E0002"
                } else if msg.contains("unused") {
                    "E0003"
                } else if msg.contains("missing return type") {
                    "E0004"
                } else if msg.contains("moved") || msg.contains("borrow") || msg.contains("ownership") {
                    "E0005"
                } else if msg.contains("effect") && (msg.contains("not declared") || msg.contains("missing")) {
                    "E0006"
                } else {
                    return;
                }
            }
        };

        match code {
            "E0001" => {
                // Type mismatch - suggest type cast or conversion
                self.add_type_mismatch_fix(uri, diagnostic, source, actions);
            }
            "E0002" => {
                // Undefined variable - suggest similar names or add declaration
                self.add_undefined_variable_fix(uri, diagnostic, source, ast, symbols, actions);
            }
            "E0003" => {
                // Unused variable - add underscore prefix
                self.add_unused_variable_fix(uri, diagnostic, actions);
            }
            "E0004" => {
                // Missing return type - add return type annotation
                self.add_missing_return_type_fix(uri, diagnostic, source, actions);
            }
            "E0005" => {
                // Ownership error - suggest clone or borrow
                self.add_ownership_fix(uri, diagnostic, source, actions);
            }
            "E0006" => {
                // Effect not declared - add effect to signature
                self.add_missing_effect_fix(uri, diagnostic, source, actions);
            }
            _ => {}
        }
    }

    /// Add fix for type mismatch (E0001)
    fn add_type_mismatch_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        // Extract expected and found types from message
        // Common patterns: "expected `X`, found `Y`" or "expected X but found Y"
        let (expected_type, found_type) = if let Some((exp, found)) = extract_type_mismatch(message) {
            (exp, found)
        } else {
            return;
        };

        let expected = TypeKind::parse(&expected_type);
        let found = TypeKind::parse(&found_type);

        // Get the expression at the diagnostic range
        let expr_text = self.get_text_in_range(source, diagnostic.range);
        if expr_text.is_empty() {
            return;
        }

        // Check if we can cast
        if found.can_cast_to(&expected) {
            if let Some(cast_expr) = found.cast_expr(&expr_text, &expected) {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: format!("Cast to `{}`", expected_type),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: diagnostic.range,
                                    new_text: cast_expr,
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            }
        }

        // For reference mismatches, suggest adding & or &!
        match (&found, &expected) {
            // Need to borrow: T -> &T
            (_, TypeKind::Ref(inner)) if **inner == found => {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: "Add `&` (borrow)".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: Range {
                                        start: diagnostic.range.start,
                                        end: diagnostic.range.start,
                                    },
                                    new_text: "&".to_string(),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            }

            // Need mutable borrow: T -> &!T (Sounio syntax)
            (_, TypeKind::RefMut(inner)) if **inner == found => {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: "Add `&!` (mutable borrow)".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: Range {
                                        start: diagnostic.range.start,
                                        end: diagnostic.range.start,
                                    },
                                    new_text: "&!".to_string(),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(false),
                    disabled: None,
                    data: None,
                }));
            }

            // Need to dereference: &T -> T
            (TypeKind::Ref(_), _) | (TypeKind::RefMut(_), _) => {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: "Add `*` (dereference)".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: Range {
                                        start: diagnostic.range.start,
                                        end: diagnostic.range.start,
                                    },
                                    new_text: "*".to_string(),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(false),
                    disabled: None,
                    data: None,
                }));
            }

            _ => {}
        }

        // Suggest .into() for convertible types
        if found != expected {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Use `.into()` to convert to `{}`", expected_type),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.end,
                                    end: diagnostic.range.end,
                                },
                                new_text: ".into()".to_string(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for undefined variable (E0002) with "Did you mean?" suggestions
    fn add_undefined_variable_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        _ast: Option<&Ast>,
        symbols: Option<&SymbolTable>,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        // Extract the undefined name from message
        let undefined_name = if let Some(name) = extract_quoted_name(message) {
            name
        } else {
            return;
        };

        // Collect names in scope for similarity matching
        let mut candidates: Vec<String> = Vec::new();

        // Get names from symbol table if available
        if let Some(symbols) = symbols {
            for sym in symbols.all_symbols() {
                candidates.push(sym.name.clone());
            }
            // Also get type names
            candidates.extend(symbols.all_type_names());
        }

        // Add built-in names
        candidates.extend(get_builtin_names().into_iter().map(|s| s.to_string()));

        // Find similar names using Levenshtein distance
        let max_distance = (undefined_name.len() / 3).max(2).min(3);
        let similar = find_similar_names(&undefined_name, &candidates, max_distance);

        // Add "Did you mean?" suggestions
        for (suggestion, distance) in similar.iter().take(3) {
            let title = if *distance == 1 {
                format!("Did you mean `{}`?", suggestion)
            } else {
                format!("Did you mean `{}`? (edit distance: {})", suggestion, distance)
            };

            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title,
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: diagnostic.range,
                                new_text: suggestion.clone(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(*distance == 1),
                disabled: None,
                data: None,
            }));
        }

        // Check if message already has a suggestion
        if let Some(suggestion) = extract_suggestion(message) {
            // Only add if not already in our list
            if !similar.iter().any(|(s, _)| s == &suggestion) {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: format!("Change to `{}`", suggestion),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: diagnostic.range,
                                    new_text: suggestion,
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            }
        }

        // Suggest adding variable declaration
        let indent = get_line_indent(source, diagnostic.range.start.line as usize);
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: format!("Declare `{}` with `let`", undefined_name),
            kind: Some(CodeActionKind::QUICKFIX),
            diagnostics: Some(vec![diagnostic.clone()]),
            edit: Some(WorkspaceEdit {
                changes: Some(
                    [(
                        uri.clone(),
                        vec![TextEdit {
                            range: Range {
                                start: Position::new(diagnostic.range.start.line, 0),
                                end: Position::new(diagnostic.range.start.line, 0),
                            },
                            new_text: format!("{}let {} = todo()\n", indent, undefined_name),
                        }],
                    )]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            }),
            command: None,
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));

        // Suggest adding mutable variable declaration
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: format!("Declare `{}` with `var` (mutable)", undefined_name),
            kind: Some(CodeActionKind::QUICKFIX),
            diagnostics: Some(vec![diagnostic.clone()]),
            edit: Some(WorkspaceEdit {
                changes: Some(
                    [(
                        uri.clone(),
                        vec![TextEdit {
                            range: Range {
                                start: Position::new(diagnostic.range.start.line, 0),
                                end: Position::new(diagnostic.range.start.line, 0),
                            },
                            new_text: format!("{}var {} = todo()\n", indent, undefined_name),
                        }],
                    )]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            }),
            command: None,
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));
    }

    /// Add fix for unused variable (E0003)
    fn add_unused_variable_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        if let Some(var_name) = extract_quoted_name(message) {
            // Add underscore prefix
            if !var_name.starts_with('_') {
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: format!("Rename to `_{}`", var_name),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: diagnostic.range,
                                    new_text: format!("_{}", var_name),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            }

            // Replace with underscore
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Replace with `_`".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: diagnostic.range,
                                new_text: "_".to_string(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for missing return type (E0004)
    fn add_missing_return_type_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Try to infer return type from context
        let inferred_types = vec!["()", "i32", "bool", "String"];

        for ty in inferred_types {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Add return type `-> {}`", ty),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.end,
                                    end: diagnostic.range.end,
                                },
                                new_text: format!(" -> {}", ty),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(ty == "()"),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for ownership errors (E0005)
    fn add_ownership_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;
        let expr_text = self.get_text_in_range(source, diagnostic.range);

        // Check what kind of ownership issue
        let is_move = message.contains("moved") || message.contains("use of moved");
        let is_borrow = message.contains("borrow") || message.contains("cannot borrow");
        let needs_mut = message.contains("mutable") || message.contains("mutably");

        if is_move {
            // Suggest .clone() to avoid move
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Add `.clone()` to avoid move".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.end,
                                    end: diagnostic.range.end,
                                },
                                new_text: ".clone()".to_string(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(true),
                disabled: None,
                data: None,
            }));

            // Suggest borrowing instead
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Use `&` (borrow instead of move)".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: diagnostic.range.start,
                                    end: diagnostic.range.start,
                                },
                                new_text: "&".to_string(),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        if is_borrow {
            if needs_mut {
                // Suggest &! (Sounio's mutable borrow syntax)
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: "Use `&!` (mutable borrow)".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: Range {
                                        start: diagnostic.range.start,
                                        end: diagnostic.range.start,
                                    },
                                    new_text: "&!".to_string(),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            } else {
                // Suggest &
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: "Use `&` (shared borrow)".to_string(),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: Some(WorkspaceEdit {
                        changes: Some(
                            [(
                                uri.clone(),
                                vec![TextEdit {
                                    range: Range {
                                        start: diagnostic.range.start,
                                        end: diagnostic.range.start,
                                    },
                                    new_text: "&".to_string(),
                                }],
                            )]
                            .into_iter()
                            .collect(),
                        ),
                        ..Default::default()
                    }),
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: None,
                }));
            }
        }

        // If the expression already has &, suggest changing to &!
        if expr_text.starts_with('&') && !expr_text.starts_with("&!") && needs_mut {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Change `&` to `&!` (mutable borrow)".to_string(),
                kind: Some(CodeActionKind::QUICKFIX),
                diagnostics: Some(vec![diagnostic.clone()]),
                edit: Some(WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: diagnostic.range,
                                new_text: format!("&!{}", &expr_text[1..]),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }),
                command: None,
                is_preferred: Some(true),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add fix for missing effect (E0006)
    fn add_missing_effect_fix(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        let message = &diagnostic.message;

        // Extract effect name(s) from message
        let effects = extract_effects_from_message(message);

        if effects.is_empty() {
            // If no specific effect found, suggest all common effects
            for effect in BUILTIN_EFFECTS {
                actions.push(self.create_effect_action(uri, diagnostic, source, effect));
            }
        } else {
            // Suggest the specific effect(s) mentioned
            for effect in &effects {
                actions.push(self.create_effect_action(uri, diagnostic, source, effect));
            }

            // If multiple effects, also suggest adding all at once
            if effects.len() > 1 {
                let effects_str = effects.join(", ");
                actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                    title: format!("Add `with {}` to function signature", effects_str),
                    kind: Some(CodeActionKind::QUICKFIX),
                    diagnostics: Some(vec![diagnostic.clone()]),
                    edit: None, // Requires finding function signature
                    command: None,
                    is_preferred: Some(true),
                    disabled: None,
                    data: Some(serde_json::json!({
                        "action": "add_effects",
                        "effects": effects
                    })),
                }));
            }
        }
    }

    /// Create a code action to add an effect to the function signature
    fn create_effect_action(
        &self,
        uri: &Url,
        diagnostic: &Diagnostic,
        source: &str,
        effect: &str,
    ) -> CodeActionOrCommand {
        // Try to find the function signature to modify
        let fn_sig_end = find_function_signature_end(source, diagnostic.range.start.line as usize);

        CodeActionOrCommand::CodeAction(CodeAction {
            title: format!("Add `with {}` to function signature", effect),
            kind: Some(CodeActionKind::QUICKFIX),
            diagnostics: Some(vec![diagnostic.clone()]),
            edit: fn_sig_end.map(|(line, col)| {
                WorkspaceEdit {
                    changes: Some(
                        [(
                            uri.clone(),
                            vec![TextEdit {
                                range: Range {
                                    start: Position::new(line as u32, col as u32),
                                    end: Position::new(line as u32, col as u32),
                                },
                                new_text: format!(" with {}", effect),
                            }],
                        )]
                        .into_iter()
                        .collect(),
                    ),
                    ..Default::default()
                }
            }),
            command: None,
            is_preferred: Some(true),
            disabled: None,
            data: Some(serde_json::json!({
                "action": "add_effect",
                "effect": effect
            })),
        })
    }

    /// Add refactoring actions based on selection
    fn add_refactoring_actions(
        &self,
        uri: &Url,
        range: Range,
        source: &str,
        _ast: Option<&Ast>,
        _symbols: Option<&SymbolTable>,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Check if there's a selection
        if range.start == range.end {
            return;
        }

        // Get selected text
        let selected = self.get_text_in_range(source, range);
        if selected.is_empty() {
            return;
        }

        let indent = get_line_indent(source, range.start.line as usize);

        // Extract to variable
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Extract to variable".to_string(),
            kind: Some(CodeActionKind::REFACTOR_EXTRACT),
            diagnostics: None,
            edit: Some(WorkspaceEdit {
                changes: Some(
                    [(
                        uri.clone(),
                        vec![
                            // Insert variable declaration before current line
                            TextEdit {
                                range: Range {
                                    start: Position::new(range.start.line, 0),
                                    end: Position::new(range.start.line, 0),
                                },
                                new_text: format!("{}let extracted = {}\n", indent, selected.trim()),
                            },
                            // Replace selection with variable name
                            TextEdit {
                                range,
                                new_text: "extracted".to_string(),
                            },
                        ],
                    )]
                    .into_iter()
                    .collect(),
                ),
                ..Default::default()
            }),
            command: None,
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));

        // Extract to function (if selection looks like a block)
        if selected.contains(';') || selected.contains('{') {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Extract to function".to_string(),
                kind: Some(CodeActionKind::REFACTOR_EXTRACT),
                diagnostics: None,
                edit: None, // Complex - would need parameter analysis
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        // Inline variable (if selection is a variable name)
        if is_identifier(&selected) {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: format!("Inline `{}`", selected.trim()),
                kind: Some(CodeActionKind::REFACTOR_INLINE),
                diagnostics: None,
                edit: None, // Would need to find definition and replace all uses
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        // Convert to string interpolation (if selection is string concatenation)
        if selected.contains("+ \"") || selected.contains("\" +") {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Convert to string interpolation".to_string(),
                kind: Some(CodeActionKind::REFACTOR_REWRITE),
                diagnostics: None,
                edit: None, // Would need string parsing
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }
    }

    /// Add source-level actions
    fn add_source_actions(
        &self,
        uri: &Url,
        source: &str,
        ast: Option<&Ast>,
        actions: &mut Vec<CodeActionOrCommand>,
    ) {
        // Organize imports
        if source.contains("import ") || source.contains("use ") {
            actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                title: "Organize imports".to_string(),
                kind: Some(CodeActionKind::SOURCE_ORGANIZE_IMPORTS),
                diagnostics: None,
                edit: None, // Would need import analysis
                command: None,
                is_preferred: Some(false),
                disabled: None,
                data: None,
            }));
        }

        // Add missing implementations (for traits/impls)
        if let Some(ast) = ast {
            for item in &ast.items {
                if let Item::Impl(impl_def) = item {
                    // Check if impl is incomplete
                    if impl_def.items.is_empty() {
                        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
                            title: "Generate missing trait methods".to_string(),
                            kind: Some(CodeActionKind::SOURCE),
                            diagnostics: None,
                            edit: None,
                            command: None,
                            is_preferred: Some(false),
                            disabled: None,
                            data: None,
                        }));
                    }
                }
            }
        }

        // Format document
        actions.push(CodeActionOrCommand::CodeAction(CodeAction {
            title: "Format document".to_string(),
            kind: Some(CodeActionKind::SOURCE),
            diagnostics: None,
            edit: None,
            command: Some(Command {
                title: "Format".to_string(),
                command: "sounio.formatDocument".to_string(),
                arguments: Some(vec![serde_json::json!(uri.as_str())]),
            }),
            is_preferred: Some(false),
            disabled: None,
            data: None,
        }));
    }

    /// Get text in a range
    fn get_text_in_range(&self, source: &str, range: Range) -> String {
        let lines: Vec<&str> = source.lines().collect();

        if range.start.line == range.end.line {
            // Single line selection
            if let Some(line) = lines.get(range.start.line as usize) {
                let start = range.start.character as usize;
                let end = range.end.character as usize;
                if start < line.len() && end <= line.len() {
                    return line[start..end].to_string();
                }
            }
        } else {
            // Multi-line selection
            let mut result = String::new();
            for (i, line) in lines.iter().enumerate() {
                let line_num = i as u32;
                if line_num == range.start.line {
                    let start = range.start.character as usize;
                    if start < line.len() {
                        result.push_str(&line[start..]);
                        result.push('\n');
                    }
                } else if line_num > range.start.line && line_num < range.end.line {
                    result.push_str(line);
                    result.push('\n');
                } else if line_num == range.end.line {
                    let end = range.end.character as usize;
                    if end <= line.len() {
                        result.push_str(&line[..end]);
                    }
                }
            }
            return result;
        }

        String::new()
    }
}

impl Default for CodeActionProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract a quoted name from a diagnostic message
fn extract_quoted_name(message: &str) -> Option<String> {
    // Look for `name` or 'name' pattern
    let patterns = [('`', '`'), ('\'', '\''), ('"', '"')];

    for (open, close) in patterns {
        if let Some(start) = message.find(open) {
            let rest = &message[start + 1..];
            if let Some(end) = rest.find(close) {
                return Some(rest[..end].to_string());
            }
        }
    }

    None
}

/// Extract a suggestion from a "did you mean" message
fn extract_suggestion(message: &str) -> Option<String> {
    if let Some(idx) = message.to_lowercase().find("did you mean") {
        let rest = &message[idx..];
        return extract_quoted_name(rest);
    }
    None
}

/// Extract expected and found types from a type mismatch message
fn extract_type_mismatch(message: &str) -> Option<(String, String)> {
    // Common patterns:
    // "expected `X`, found `Y`"
    // "expected X but found Y"
    // "type mismatch: expected `X`, found `Y`"

    let msg_lower = message.to_lowercase();

    // Find "expected"
    let exp_idx = msg_lower.find("expected")?;
    let after_exp = &message[exp_idx + 8..];

    // Extract expected type
    let expected = if after_exp.trim().starts_with('`') {
        extract_quoted_name(after_exp)?
    } else {
        // Non-quoted: take until comma or "but" or "found"
        let end = after_exp.find(',')
            .or_else(|| after_exp.to_lowercase().find(" but"))
            .or_else(|| after_exp.to_lowercase().find(" found"))
            .unwrap_or(after_exp.len());
        after_exp[..end].trim().to_string()
    };

    // Find "found"
    let found_idx = msg_lower.find("found")?;
    let after_found = &message[found_idx + 5..];

    // Extract found type
    let found = if after_found.trim().starts_with('`') {
        extract_quoted_name(after_found)?
    } else {
        after_found.trim().split_whitespace().next()?.to_string()
    };

    Some((expected, found))
}

/// Extract effect names from a diagnostic message
fn extract_effects_from_message(message: &str) -> Vec<String> {
    let mut effects = Vec::new();

    // Look for quoted effect names
    if let Some(effect) = extract_quoted_name(message) {
        if BUILTIN_EFFECTS.contains(&effect.as_str()) {
            effects.push(effect);
        }
    }

    // Also look for unquoted effect names in the message
    for effect in BUILTIN_EFFECTS {
        if message.contains(effect) && !effects.contains(&effect.to_string()) {
            effects.push(effect.to_string());
        }
    }

    effects
}

/// Check if a string is a valid identifier
fn is_identifier(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }

    let mut chars = s.chars();
    if let Some(first) = chars.next() {
        if !first.is_alphabetic() && first != '_' {
            return false;
        }
    }

    chars.all(|c| c.is_alphanumeric() || c == '_')
}

/// Get the indentation of a line
fn get_line_indent(source: &str, line_num: usize) -> String {
    source
        .lines()
        .nth(line_num)
        .map(|line| {
            let trimmed = line.trim_start();
            line[..line.len() - trimmed.len()].to_string()
        })
        .unwrap_or_default()
}

/// Find the end of a function signature (before the opening brace)
fn find_function_signature_end(source: &str, start_line: usize) -> Option<(usize, usize)> {
    let lines: Vec<&str> = source.lines().collect();

    // Search backward from the diagnostic line to find the function definition
    for line_num in (0..=start_line).rev() {
        if let Some(line) = lines.get(line_num) {
            // Look for "fn " keyword
            if let Some(fn_pos) = line.find("fn ") {
                // Find the opening brace or arrow
                for search_line in line_num..lines.len().min(line_num + 5) {
                    if let Some(search_line_content) = lines.get(search_line) {
                        // Check for existing "with" clause
                        if search_line_content.contains(" with ") {
                            // Already has effects, find the end of the with clause
                            if let Some(brace_pos) = search_line_content.find('{') {
                                return Some((search_line, brace_pos - 1));
                            }
                        }
                        // Look for the opening brace
                        if let Some(brace_pos) = search_line_content.find('{') {
                            // Insert before the brace
                            return Some((search_line, brace_pos));
                        }
                    }
                }
            }
        }
    }

    None
}

/// Get built-in names for similarity matching
fn get_builtin_names() -> Vec<&'static str> {
    vec![
        // Types
        "i8", "i16", "i32", "i64", "i128", "isize",
        "u8", "u16", "u32", "u64", "u128", "usize",
        "f32", "f64", "bool", "char", "String", "str",
        "Vec", "HashMap", "HashSet", "Option", "Result",
        // Functions
        "print", "println", "dbg", "panic", "assert", "assert_eq",
        "len", "sqrt", "abs", "sin", "cos", "tan", "exp", "log",
        "min", "max", "floor", "ceil", "round",
        // Option/Result variants
        "Some", "None", "Ok", "Err",
        // Effects
        "IO", "Mut", "Alloc", "Panic", "Async", "GPU", "Prob", "Div",
        // Linear algebra
        "vec2", "vec3", "vec4", "mat2", "mat3", "mat4", "quat",
        "dot", "cross", "normalize", "length",
        // Probabilistic
        "sample", "observe", "infer",
        "Normal", "Uniform", "Beta", "Gamma",
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = CodeActionProvider::new();
        assert!(provider.enable_quick_fixes);
        assert!(provider.enable_refactorings);
        assert!(provider.enable_source_actions);
    }

    #[test]
    fn test_levenshtein_distance() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("abc", "abd"), 1);
        assert_eq!(levenshtein_distance("abc", "adc"), 1);
        assert_eq!(levenshtein_distance("abc", "abcd"), 1);
    }

    #[test]
    fn test_normalized_levenshtein() {
        let dist = normalized_levenshtein("Concentration", "Concentraiton");
        assert!(dist < 0.2); // Close strings

        let dist = normalized_levenshtein("Absorption", "Metabolism");
        assert!(dist > 0.5); // Different strings

        assert_eq!(normalized_levenshtein("abc", "abc"), 0.0);
    }

    #[test]
    fn test_find_similar_names() {
        let candidates = vec![
            "print".to_string(),
            "println".to_string(),
            "panic".to_string(),
            "assert".to_string(),
        ];

        let similar = find_similar_names("prit", &candidates, 2);
        assert!(!similar.is_empty());
        assert_eq!(similar[0].0, "print");
        assert_eq!(similar[0].1, 1);
    }

    #[test]
    fn test_similarity_score() {
        assert_eq!(similarity_score("abc", "abc"), 1.0);
        assert!(similarity_score("abc", "abd") > 0.5);
        assert!(similarity_score("abc", "xyz") < 0.5);
    }

    #[test]
    fn test_extract_quoted_name() {
        assert_eq!(
            extract_quoted_name("undefined variable `foo`"),
            Some("foo".to_string())
        );
        assert_eq!(
            extract_quoted_name("unknown type 'Bar'"),
            Some("Bar".to_string())
        );
        assert_eq!(extract_quoted_name("no quotes here"), None);
    }

    #[test]
    fn test_extract_suggestion() {
        assert_eq!(
            extract_suggestion("undefined variable `foo`, did you mean `for`?"),
            Some("for".to_string())
        );
        assert_eq!(extract_suggestion("no suggestion"), None);
    }

    #[test]
    fn test_extract_type_mismatch() {
        let result = extract_type_mismatch("expected `i32`, found `i64`");
        assert_eq!(result, Some(("i32".to_string(), "i64".to_string())));

        let result = extract_type_mismatch("type mismatch: expected `String`, found `&str`");
        assert_eq!(result, Some(("String".to_string(), "&str".to_string())));
    }

    #[test]
    fn test_is_identifier() {
        assert!(is_identifier("foo"));
        assert!(is_identifier("_bar"));
        assert!(is_identifier("baz123"));
        assert!(!is_identifier("123abc"));
        assert!(!is_identifier("foo bar"));
        assert!(!is_identifier(""));
    }

    #[test]
    fn test_type_kind_parse() {
        assert_eq!(TypeKind::parse("i32"), TypeKind::Int { bits: 32, signed: true });
        assert_eq!(TypeKind::parse("f64"), TypeKind::Float { bits: 64 });
        assert_eq!(TypeKind::parse("bool"), TypeKind::Bool);

        // Sounio mutable reference syntax
        assert_eq!(
            TypeKind::parse("&!i32"),
            TypeKind::RefMut(Box::new(TypeKind::Int { bits: 32, signed: true }))
        );

        assert_eq!(
            TypeKind::parse("&i32"),
            TypeKind::Ref(Box::new(TypeKind::Int { bits: 32, signed: true }))
        );
    }

    #[test]
    fn test_type_kind_can_cast() {
        let i32_ty = TypeKind::Int { bits: 32, signed: true };
        let i64_ty = TypeKind::Int { bits: 64, signed: true };
        let f64_ty = TypeKind::Float { bits: 64 };

        assert!(i32_ty.can_cast_to(&i64_ty));
        assert!(i32_ty.can_cast_to(&f64_ty));
        assert!(f64_ty.can_cast_to(&i32_ty));
    }

    #[test]
    fn test_get_text_in_range_single_line() {
        let provider = CodeActionProvider::new();
        let source = "let x = 42;";
        let range = Range {
            start: Position::new(0, 4),
            end: Position::new(0, 5),
        };
        assert_eq!(provider.get_text_in_range(source, range), "x");
    }

    #[test]
    fn test_get_text_in_range_multi_line() {
        let provider = CodeActionProvider::new();
        let source = "let x = 42;\nlet y = 10;";
        let range = Range {
            start: Position::new(0, 8),
            end: Position::new(1, 3),
        };
        let text = provider.get_text_in_range(source, range);
        assert!(text.contains("42"));
        assert!(text.contains("let"));
    }

    #[test]
    fn test_code_actions_for_undefined_variable() {
        let provider = CodeActionProvider::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let diagnostic = Diagnostic {
            range: Range {
                start: Position::new(0, 0),
                end: Position::new(0, 3),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("E0002".to_string())),
            message: "undefined variable `foo`".to_string(),
            ..Default::default()
        };

        let actions =
            provider.code_actions(&uri, diagnostic.range, &[diagnostic], "foo", None, None);
        assert!(!actions.is_empty());

        // Should have "did you mean" suggestions
        let titles: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => Some(ca.title.as_str()),
                _ => None,
            })
            .collect();

        // Should have declaration suggestions
        assert!(titles.iter().any(|t| t.contains("Declare")));
    }

    #[test]
    fn test_code_actions_for_type_mismatch() {
        let provider = CodeActionProvider::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let diagnostic = Diagnostic {
            range: Range {
                start: Position::new(0, 8),
                end: Position::new(0, 10),
            },
            severity: Some(DiagnosticSeverity::ERROR),
            code: Some(NumberOrString::String("E0001".to_string())),
            message: "type mismatch: expected `i64`, found `i32`".to_string(),
            ..Default::default()
        };

        let source = "let x = 42";
        let actions =
            provider.code_actions(&uri, diagnostic.range, &[diagnostic], source, None, None);

        // Should have cast suggestion
        let titles: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => Some(ca.title.as_str()),
                _ => None,
            })
            .collect();

        assert!(titles.iter().any(|t| t.contains("Cast") || t.contains("into")));
    }

    #[test]
    fn test_refactoring_actions() {
        let provider = CodeActionProvider::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let source = "let x = 1 + 2 + 3;";
        let range = Range {
            start: Position::new(0, 8),
            end: Position::new(0, 17),
        };

        let actions = provider.code_actions(&uri, range, &[], source, None, None);
        // Should have extract to variable action
        let titles: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                CodeActionOrCommand::CodeAction(ca) => Some(ca.title.as_str()),
                _ => None,
            })
            .collect();
        assert!(titles.contains(&"Extract to variable"));
    }

    #[test]
    fn test_extract_effects_from_message() {
        let effects = extract_effects_from_message("effect `IO` not declared in function signature");
        assert!(effects.contains(&"IO".to_string()));

        let effects = extract_effects_from_message("function uses Prob effect without declaring it");
        assert!(effects.contains(&"Prob".to_string()));
    }

    #[test]
    fn test_get_line_indent() {
        let source = "fn foo() {\n    let x = 1;\n        let y = 2;\n}";
        assert_eq!(get_line_indent(source, 0), "");
        assert_eq!(get_line_indent(source, 1), "    ");
        assert_eq!(get_line_indent(source, 2), "        ");
    }
}
