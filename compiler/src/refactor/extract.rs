//! Extract Refactoring
//!
//! Provides extraction refactorings:
//! - Extract Function: Extract selected code into a new function
//! - Extract Variable: Extract expression into a local variable
//! - Extract Constant: Extract literal into a named constant

use super::{RefactoringContext, RefactoringError, TextEdit, WorkspaceEdit, validate_new_name};
use crate::ast::Ast;
use std::collections::HashSet;

/// Extract function refactoring
pub struct ExtractFunction {
    /// Name for the new function
    pub function_name: String,
    /// Selection range (byte offsets)
    pub selection: std::ops::Range<usize>,
    /// Whether the function should be public
    pub is_public: bool,
    /// Whether to add doc comment
    pub add_doc_comment: bool,
}

impl ExtractFunction {
    /// Create a new extract function operation
    pub fn new(function_name: impl Into<String>, selection: std::ops::Range<usize>) -> Self {
        Self {
            function_name: function_name.into(),
            selection,
            is_public: false,
            add_doc_comment: false,
        }
    }

    /// Make the extracted function public
    pub fn public(mut self) -> Self {
        self.is_public = true;
        self
    }

    /// Add a doc comment to the extracted function
    pub fn with_doc_comment(mut self) -> Self {
        self.add_doc_comment = true;
        self
    }

    /// Execute the extraction
    pub fn execute(&self, ctx: &RefactoringContext) -> Result<ExtractResult, RefactoringError> {
        // Validate function name
        validate_new_name(&self.function_name)?;

        let source = ctx
            .current_source()
            .ok_or_else(|| RefactoringError::SourceNotFound(ctx.current_file.clone()))?;
        let ast = ctx
            .current_ast()
            .ok_or_else(|| RefactoringError::ModuleNotFound(ctx.current_file.clone()))?;

        // Validate selection
        if self.selection.start >= self.selection.end || self.selection.end > source.len() {
            return Err(RefactoringError::InvalidSelection {
                start: self.selection.start,
                end: self.selection.end,
            });
        }

        // Get selected code
        let selected_code = &source[self.selection.clone()];

        // Analyze the selected code to find:
        // - Variables used (need to be parameters)
        // - Variables defined (might need to be returned)
        // - Return type
        let analysis = self.analyze_selection(ast, source, &self.selection);

        // Generate the new function
        let new_function = self.generate_function(&analysis, selected_code);

        // Generate the replacement call
        let replacement_call = self.generate_call(&analysis);

        // Find insertion point (after containing function or at module level)
        let insertion_point = self.find_insertion_point(source, &self.selection);

        // Create edits
        let mut workspace_edit = WorkspaceEdit::new();

        // 1. Replace selected code with function call
        workspace_edit.add_edit(
            &ctx.current_file,
            TextEdit::replace(self.selection.clone(), replacement_call.clone()),
        );

        // 2. Insert new function
        workspace_edit.add_edit(
            &ctx.current_file,
            TextEdit::insert(insertion_point, format!("\n\n{}", new_function)),
        );

        Ok(ExtractResult {
            kind: ExtractKind::Function,
            name: self.function_name.clone(),
            generated_code: new_function,
            replacement_code: replacement_call,
            edits: workspace_edit,
            parameters: analysis.free_variables.to_vec(),
            return_type: analysis.inferred_return_type,
        })
    }

    /// Analyze the selection to determine parameters and return type
    fn analyze_selection(
        &self,
        _ast: &Ast,
        source: &str,
        selection: &std::ops::Range<usize>,
    ) -> SelectionAnalysis {
        let analyzer = SelectionAnalyzer::new(source, selection.clone());

        // Simple heuristic analysis based on text
        let selected = &source[selection.clone()];

        // Find identifiers that are used but not defined in selection
        let mut free_vars = HashSet::new();
        let mut defined_vars = HashSet::new();

        // Very simple lexical analysis
        let tokens: Vec<&str> = selected
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .filter(|s| !s.is_empty())
            .collect();

        let keywords = [
            "let", "var", "const", "fn", "if", "else", "match", "for", "while", "loop", "break",
            "continue", "return", "true", "false", "self",
        ];

        for token in tokens {
            if keywords.contains(&token) {
                continue;
            }
            // Check if it looks like a variable (starts with lowercase)
            if token
                .chars()
                .next()
                .map(|c| c.is_lowercase())
                .unwrap_or(false)
            {
                free_vars.insert(token.to_string());
            }
        }

        // Check for let bindings in selection
        for line in selected.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("let ") || trimmed.starts_with("var ") {
                // Extract variable name
                let parts: Vec<&str> = trimmed.split_whitespace().collect();
                if parts.len() >= 2 {
                    let name = parts[1].trim_end_matches([':', '=']);
                    defined_vars.insert(name.to_string());
                    free_vars.remove(name);
                }
            }
        }

        // Infer return type (simple heuristic)
        let return_type = if selected.contains("return ") {
            Some("_".to_string()) // Unknown, use inference
        } else if selected.ends_with(';') || selected.lines().count() > 1 {
            Some("()".to_string()) // Unit type
        } else {
            Some("_".to_string()) // Expression, infer type
        };

        SelectionAnalysis {
            free_variables: free_vars.into_iter().collect(),
            defined_variables: defined_vars.into_iter().collect(),
            inferred_return_type: return_type,
            needs_mutable: selected.contains("var ") || selected.contains("&!"),
            has_return: selected.contains("return "),
            has_break: selected.contains("break ") || selected.contains("continue "),
        }
    }

    /// Generate the new function code
    fn generate_function(&self, analysis: &SelectionAnalysis, body: &str) -> String {
        let mut result = String::new();

        // Doc comment
        if self.add_doc_comment {
            result.push_str(&format!(
                "/// TODO: Add documentation for {}\n",
                self.function_name
            ));
        }

        // Visibility
        if self.is_public {
            result.push_str("pub ");
        }

        // Function signature
        result.push_str("fn ");
        result.push_str(&self.function_name);
        result.push('(');

        // Parameters
        let params: Vec<String> = analysis
            .free_variables
            .iter()
            .map(|v| format!("{}: _", v))
            .collect();
        result.push_str(&params.join(", "));

        result.push(')');

        // Return type
        if let Some(ref ret_type) = analysis.inferred_return_type
            && ret_type != "()"
        {
            result.push_str(" -> ");
            result.push_str(ret_type);
        }

        // Body
        result.push_str(" {\n");

        // Indent body
        for line in body.lines() {
            result.push_str("    ");
            result.push_str(line);
            result.push('\n');
        }

        // Ensure there's a return if needed
        if !analysis.has_return && analysis.inferred_return_type.as_deref() != Some("()") {
            // The last expression should be the return value
        }

        result.push('}');

        result
    }

    /// Generate the replacement call
    fn generate_call(&self, analysis: &SelectionAnalysis) -> String {
        let args = analysis.free_variables.join(", ");
        format!("{}({})", self.function_name, args)
    }

    /// Find where to insert the new function
    fn find_insertion_point(&self, source: &str, selection: &std::ops::Range<usize>) -> usize {
        // Find the end of the containing function
        // Simple heuristic: find matching brace
        let before = &source[..selection.start];

        // Count braces to find function end
        let mut brace_count = 0;
        let mut in_function = false;
        let mut function_end = source.len();

        for (i, c) in source.char_indices() {
            match c {
                '{' => {
                    brace_count += 1;
                    if i < selection.start {
                        in_function = true;
                    }
                }
                '}' => {
                    brace_count -= 1;
                    if in_function && brace_count == 0 && i > selection.end {
                        function_end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }

        function_end
    }
}

/// Extract variable refactoring
pub struct ExtractVariable {
    /// Name for the new variable
    pub variable_name: String,
    /// Selection range (byte offsets)
    pub selection: std::ops::Range<usize>,
    /// Whether to use `let` (false) or `var` (true)
    pub mutable: bool,
}

impl ExtractVariable {
    /// Create a new extract variable operation
    pub fn new(variable_name: impl Into<String>, selection: std::ops::Range<usize>) -> Self {
        Self {
            variable_name: variable_name.into(),
            selection,
            mutable: false,
        }
    }

    /// Make the variable mutable
    pub fn mutable(mut self) -> Self {
        self.mutable = true;
        self
    }

    /// Execute the extraction
    pub fn execute(&self, ctx: &RefactoringContext) -> Result<ExtractResult, RefactoringError> {
        // Validate variable name
        validate_new_name(&self.variable_name)?;

        let source = ctx
            .current_source()
            .ok_or_else(|| RefactoringError::SourceNotFound(ctx.current_file.clone()))?;

        // Validate selection
        if self.selection.start >= self.selection.end || self.selection.end > source.len() {
            return Err(RefactoringError::InvalidSelection {
                start: self.selection.start,
                end: self.selection.end,
            });
        }

        // Get selected expression
        let selected_expr = &source[self.selection.clone()];

        // Generate variable declaration
        let keyword = if self.mutable { "var" } else { "let" };
        let declaration = format!(
            "{} {} = {};\n",
            keyword,
            self.variable_name,
            selected_expr.trim()
        );

        // Find where to insert the declaration (start of statement)
        let insertion_point = self.find_statement_start(source, self.selection.start);

        // Calculate indentation
        let indent = self.get_indentation(source, insertion_point);
        let indented_declaration = format!("{}{}", indent, declaration);

        // Create edits
        let mut workspace_edit = WorkspaceEdit::new();

        // 1. Insert variable declaration
        workspace_edit.add_edit(
            &ctx.current_file,
            TextEdit::insert(insertion_point, indented_declaration.clone()),
        );

        // Adjust selection range due to insertion
        let adjusted_selection = std::ops::Range {
            start: self.selection.start + indented_declaration.len(),
            end: self.selection.end + indented_declaration.len(),
        };

        // 2. Replace selected expression with variable name
        workspace_edit.add_edit(
            &ctx.current_file,
            TextEdit::replace(adjusted_selection, self.variable_name.clone()),
        );

        Ok(ExtractResult {
            kind: ExtractKind::Variable,
            name: self.variable_name.clone(),
            generated_code: declaration,
            replacement_code: self.variable_name.clone(),
            edits: workspace_edit,
            parameters: vec![],
            return_type: None,
        })
    }

    /// Find the start of the containing statement
    fn find_statement_start(&self, source: &str, position: usize) -> usize {
        let before = &source[..position];

        // Find previous newline
        if let Some(newline_pos) = before.rfind('\n') {
            newline_pos + 1
        } else {
            0
        }
    }

    /// Get the indentation at a position
    fn get_indentation(&self, source: &str, position: usize) -> String {
        let line_start = position;
        let after = &source[line_start..];

        let indent: String = after
            .chars()
            .take_while(|c| c.is_whitespace() && *c != '\n')
            .collect();

        indent
    }
}

/// Result of an extraction operation
#[derive(Debug, Clone)]
pub struct ExtractResult {
    /// Kind of extraction performed
    pub kind: ExtractKind,
    /// Name of the extracted item
    pub name: String,
    /// The generated code (function, variable declaration, etc.)
    pub generated_code: String,
    /// Code that replaces the selection
    pub replacement_code: String,
    /// Edits to apply
    pub edits: WorkspaceEdit,
    /// Parameters for extracted function
    pub parameters: Vec<String>,
    /// Inferred return type
    pub return_type: Option<String>,
}

/// Kind of extraction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExtractKind {
    Function,
    Variable,
    Constant,
}

impl std::fmt::Display for ExtractKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractKind::Function => write!(f, "function"),
            ExtractKind::Variable => write!(f, "variable"),
            ExtractKind::Constant => write!(f, "constant"),
        }
    }
}

/// Analysis of selected code for extraction
#[derive(Debug)]
struct SelectionAnalysis {
    /// Variables used but not defined in selection
    free_variables: Vec<String>,
    /// Variables defined in selection
    defined_variables: Vec<String>,
    /// Inferred return type
    inferred_return_type: Option<String>,
    /// Whether the selection needs mutable access
    needs_mutable: bool,
    /// Whether selection contains return
    has_return: bool,
    /// Whether selection contains break/continue
    has_break: bool,
}

/// Helper for analyzing selections
struct SelectionAnalyzer {
    source: String,
    selection: std::ops::Range<usize>,
    used_variables: HashSet<String>,
    defined_variables: HashSet<String>,
}

impl SelectionAnalyzer {
    fn new(source: &str, selection: std::ops::Range<usize>) -> Self {
        Self {
            source: source.to_string(),
            selection,
            used_variables: HashSet::new(),
            defined_variables: HashSet::new(),
        }
    }
}

/// Extract constant refactoring
pub struct ExtractConstant {
    /// Name for the constant
    pub constant_name: String,
    /// Selection range
    pub selection: std::ops::Range<usize>,
    /// Whether to make it public
    pub is_public: bool,
}

impl ExtractConstant {
    /// Create a new extract constant operation
    pub fn new(constant_name: impl Into<String>, selection: std::ops::Range<usize>) -> Self {
        Self {
            constant_name: constant_name.into(),
            selection,
            is_public: false,
        }
    }

    /// Make the constant public
    pub fn public(mut self) -> Self {
        self.is_public = true;
        self
    }

    /// Execute the extraction
    pub fn execute(&self, ctx: &RefactoringContext) -> Result<ExtractResult, RefactoringError> {
        // Validate constant name (should be SCREAMING_SNAKE_CASE)
        validate_new_name(&self.constant_name)?;

        let source = ctx
            .current_source()
            .ok_or_else(|| RefactoringError::SourceNotFound(ctx.current_file.clone()))?;

        // Validate selection
        if self.selection.start >= self.selection.end || self.selection.end > source.len() {
            return Err(RefactoringError::InvalidSelection {
                start: self.selection.start,
                end: self.selection.end,
            });
        }

        // Get selected literal
        let selected = source[self.selection.clone()].trim();

        // Infer type from literal
        let type_annotation = infer_literal_type(selected);

        // Generate constant declaration
        let visibility = if self.is_public { "pub " } else { "" };
        let declaration = format!(
            "{}const {}: {} = {};\n\n",
            visibility, self.constant_name, type_annotation, selected
        );

        // Find module-level insertion point
        let insertion_point = 0; // For simplicity, insert at top

        // Create edits
        let mut workspace_edit = WorkspaceEdit::new();

        // 1. Insert constant at module level
        workspace_edit.add_edit(
            &ctx.current_file,
            TextEdit::insert(insertion_point, declaration.clone()),
        );

        // Adjust selection due to insertion
        let adjusted_selection = std::ops::Range {
            start: self.selection.start + declaration.len(),
            end: self.selection.end + declaration.len(),
        };

        // 2. Replace selected literal with constant name
        workspace_edit.add_edit(
            &ctx.current_file,
            TextEdit::replace(adjusted_selection, self.constant_name.clone()),
        );

        Ok(ExtractResult {
            kind: ExtractKind::Constant,
            name: self.constant_name.clone(),
            generated_code: declaration,
            replacement_code: self.constant_name.clone(),
            edits: workspace_edit,
            parameters: vec![],
            return_type: Some(type_annotation),
        })
    }
}

/// Infer type from a literal value
fn infer_literal_type(literal: &str) -> String {
    if literal == "true" || literal == "false" {
        "bool".to_string()
    } else if literal.starts_with('"') {
        "string".to_string()
    } else if literal.starts_with('\'') {
        "char".to_string()
    } else if literal.contains('.') {
        "f64".to_string()
    } else if literal.parse::<i64>().is_ok() {
        "i64".to_string()
    } else {
        "_".to_string() // Unknown, use inference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_function_new() {
        let ef = ExtractFunction::new("helper", 10..20);
        assert_eq!(ef.function_name, "helper");
        assert_eq!(ef.selection, 10..20);
        assert!(!ef.is_public);
    }

    #[test]
    fn test_extract_function_public() {
        let ef = ExtractFunction::new("helper", 10..20).public();
        assert!(ef.is_public);
    }

    #[test]
    fn test_extract_variable_new() {
        let ev = ExtractVariable::new("result", 10..20);
        assert_eq!(ev.variable_name, "result");
        assert!(!ev.mutable);
    }

    #[test]
    fn test_extract_variable_mutable() {
        let ev = ExtractVariable::new("counter", 10..20).mutable();
        assert!(ev.mutable);
    }

    #[test]
    fn test_extract_constant_new() {
        let ec = ExtractConstant::new("MAX_SIZE", 10..20);
        assert_eq!(ec.constant_name, "MAX_SIZE");
        assert!(!ec.is_public);
    }

    #[test]
    fn test_infer_literal_type() {
        assert_eq!(infer_literal_type("true"), "bool");
        assert_eq!(infer_literal_type("false"), "bool");
        assert_eq!(infer_literal_type("\"hello\""), "string");
        assert_eq!(infer_literal_type("'x'"), "char");
        assert_eq!(infer_literal_type("3.14"), "f64");
        assert_eq!(infer_literal_type("42"), "i64");
    }

    #[test]
    fn test_extract_kind_display() {
        assert_eq!(format!("{}", ExtractKind::Function), "function");
        assert_eq!(format!("{}", ExtractKind::Variable), "variable");
        assert_eq!(format!("{}", ExtractKind::Constant), "constant");
    }
}
