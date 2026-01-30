//! Style lints
//!
//! These lints detect style issues in Sounio code, including naming
//! conventions, unused declarations, and formatting issues.
//!
//! # Naming Conventions in Sounio
//!
//! - Functions/methods: `snake_case`
//! - Variables: `snake_case`
//! - Types (structs, enums, traits): `PascalCase`
//! - Constants: `SCREAMING_SNAKE_CASE`
//! - Modules: `snake_case`
//!
//! # Example
//!
//! ```sio
//! // Good naming
//! fn calculate_dose(patient_weight: kg) -> mg { ... }
//! struct PatientRecord { ... }
//! const MAX_DOSE: mg = 500.0
//!
//! // Bad naming (will trigger warnings)
//! fn CalculateDose(...) { ... }  // should be snake_case
//! struct patient_record { ... }  // should be PascalCase
//! ```

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::ast::{Expr, Item, Pattern, Stmt};
use crate::common::Span;
use crate::lint::{Applicability, Edit, Lint, LintCategory, LintContext, LintLevel, Suggestion};
use std::collections::HashSet;

// =============================================================================
// Unused Variable
// =============================================================================

/// Detects variables that are declared but never used
///
/// Variables that are never used after declaration are usually a sign of
/// incomplete code or dead code. This lint suggests prefixing with `_` to
/// indicate intentionally unused variables.
pub struct UnusedVariable;

impl Lint for UnusedVariable {
    fn id(&self) -> &'static str {
        "unused_variable"
    }

    fn name(&self) -> &'static str {
        "Unused Variable"
    }

    fn description(&self) -> &'static str {
        "Warns about variables that are declared but never used."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Style
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            // Collect all declared variables
            let mut declared: Vec<(String, Span)> = Vec::new();
            let mut used: HashSet<String> = HashSet::new();

            // Add parameters as declared
            for param in &f.params {
                if let Pattern::Binding { name, .. } = &param.pattern {
                    if !name.starts_with('_') {
                        declared.push((name.clone(), Span::dummy()));
                    }
                }
            }

            // Collect declarations and usages from body
            collect_declarations_and_usages(&f.body, &mut declared, &mut used);

            // Report unused variables
            for (name, span) in declared {
                if !used.contains(&name) && !name.starts_with('_') {
                    ctx.report_with_suggestion(
                        self,
                        format!("unused variable: `{}`", name),
                        span,
                        Suggestion {
                            message: format!(
                                "if this is intentional, prefix with underscore: `_{}`",
                                name
                            ),
                            edits: vec![Edit {
                                span: Span {
                                    start: span.start,
                                    end: span.start,
                                },
                                replacement: "_".to_string(),
                            }],
                            applicability: Applicability::MachineApplicable,
                        },
                    );
                }
            }
        }
    }
}

/// Collect variable declarations and usages from a block
fn collect_declarations_and_usages(
    block: &crate::ast::Block,
    declared: &mut Vec<(String, Span)>,
    used: &mut HashSet<String>,
) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Let { pattern, value, .. } => {
                // Register declaration
                if let Pattern::Binding { name, .. } = pattern {
                    if !name.starts_with('_') {
                        declared.push((name.clone(), Span::dummy()));
                    }
                }

                // Check value for usages
                if let Some(expr) = value {
                    collect_usages_expr(expr, used);
                }
            }
            Stmt::Expr { expr, .. } => {
                collect_usages_expr(expr, used);
            }
            Stmt::Assign { target, value, .. } => {
                collect_usages_expr(target, used);
                collect_usages_expr(value, used);
            }
            _ => {}
        }
    }
}

/// Collect variable usages from an expression
fn collect_usages_expr(expr: &Expr, used: &mut HashSet<String>) {
    match expr {
        Expr::Path { path, .. } => {
            if !path.segments.is_empty() {
                used.insert(path.segments[0].clone());
            }
        }
        Expr::Call { callee, args, .. } => {
            collect_usages_expr(callee, used);
            for arg in args {
                collect_usages_expr(arg, used);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            collect_usages_expr(receiver, used);
            for arg in args {
                collect_usages_expr(arg, used);
            }
        }
        Expr::Binary { left, right, .. } => {
            collect_usages_expr(left, used);
            collect_usages_expr(right, used);
        }
        Expr::Unary { expr, .. } => {
            collect_usages_expr(expr, used);
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            collect_usages_expr(condition, used);
            collect_declarations_and_usages(then_branch, &mut vec![], used);
            if let Some(else_expr) = else_branch {
                collect_usages_expr(else_expr, used);
            }
        }
        Expr::Block { block, .. } => {
            collect_declarations_and_usages(block, &mut vec![], used);
        }
        Expr::Field { base, .. } => {
            collect_usages_expr(base, used);
        }
        Expr::Index { base, index, .. } => {
            collect_usages_expr(base, used);
            collect_usages_expr(index, used);
        }
        Expr::Closure { body, .. } => {
            collect_usages_expr(body, used);
        }
        Expr::Loop { body, .. } | Expr::While { body, .. } | Expr::For { body, .. } => {
            collect_declarations_and_usages(body, &mut vec![], used);
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            collect_usages_expr(scrutinee, used);
            for arm in arms {
                collect_usages_expr(&arm.body, used);
            }
        }
        Expr::Return { value, .. } => {
            if let Some(v) = value {
                collect_usages_expr(v, used);
            }
        }
        Expr::Array { elements, .. } => {
            for elem in elements {
                collect_usages_expr(elem, used);
            }
        }
        Expr::Tuple { elements, .. } => {
            for elem in elements {
                collect_usages_expr(elem, used);
            }
        }
        _ => {}
    }
}

// =============================================================================
// Unused Import
// =============================================================================

/// Detects imports that are declared but never used
pub struct UnusedImport;

impl Lint for UnusedImport {
    fn id(&self) -> &'static str {
        "unused_import"
    }

    fn name(&self) -> &'static str {
        "Unused Import"
    }

    fn description(&self) -> &'static str {
        "Warns about imports that are declared but never used in the module."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Style
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        // Check for import statements that are not referenced
        if let Item::Import(i) = item {
            // Get the imported name(s)
            let imported_names = extract_imported_names(&i.path);

            for name in imported_names {
                // Track this import for later usage checking
                ctx.define_name(&name, i.span, crate::lint::NameKind::Variable);
            }
        }
    }
}

/// Extract imported names from a use path
fn extract_imported_names(path: &crate::ast::Path) -> Vec<String> {
    let mut names = Vec::new();

    // Get the last segment as the imported name
    if let Some(last) = path.segments.last() {
        names.push(last.clone());
    }

    names
}

// =============================================================================
// Naming Convention
// =============================================================================

/// Detects violations of naming conventions
///
/// Sounio follows these naming conventions:
/// - Functions, methods, variables, parameters: `snake_case`
/// - Types (structs, enums, traits, type aliases): `PascalCase`
/// - Constants: `SCREAMING_SNAKE_CASE`
pub struct NamingConvention;

impl Lint for NamingConvention {
    fn id(&self) -> &'static str {
        "naming_convention"
    }

    fn name(&self) -> &'static str {
        "Naming Convention"
    }

    fn description(&self) -> &'static str {
        "Warns when names don't follow Sounio naming conventions."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Naming
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        match item {
            Item::Function(f) => {
                // Functions should be snake_case
                if !is_snake_case(&f.name) {
                    let suggested = to_snake_case(&f.name);
                    ctx.report_with_suggestion(
                        self,
                        format!("function `{}` should use snake_case", f.name),
                        f.span,
                        Suggestion {
                            message: format!("rename to `{}`", suggested),
                            edits: vec![],
                            applicability: Applicability::MaybeIncorrect,
                        },
                    );
                }

                // Check parameters
                for param in &f.params {
                    if let Pattern::Binding { name, .. } = &param.pattern {
                        if !is_snake_case(name) && !name.starts_with('_') {
                            let suggested = to_snake_case(name);
                            ctx.report_with_help(
                                self,
                                format!("parameter `{}` should use snake_case", name),
                                Span::dummy(),
                                format!("rename to `{}`", suggested),
                            );
                        }
                    }
                }
            }
            Item::Struct(s) => {
                // Structs should be PascalCase
                if !is_pascal_case(&s.name) {
                    let suggested = to_pascal_case(&s.name);
                    ctx.report_with_suggestion(
                        self,
                        format!("struct `{}` should use PascalCase", s.name),
                        s.span,
                        Suggestion {
                            message: format!("rename to `{}`", suggested),
                            edits: vec![],
                            applicability: Applicability::MaybeIncorrect,
                        },
                    );
                }

                // Check fields
                for field in &s.fields {
                    if !is_snake_case(&field.name) {
                        let suggested = to_snake_case(&field.name);
                        ctx.report_with_help(
                            self,
                            format!("field `{}` should use snake_case", field.name),
                            Span::dummy(),
                            format!("rename to `{}`", suggested),
                        );
                    }
                }
            }
            Item::Enum(e) => {
                // Enums should be PascalCase
                if !is_pascal_case(&e.name) {
                    let suggested = to_pascal_case(&e.name);
                    ctx.report_with_help(
                        self,
                        format!("enum `{}` should use PascalCase", e.name),
                        e.span,
                        format!("rename to `{}`", suggested),
                    );
                }

                // Variants should be PascalCase
                for variant in &e.variants {
                    if !is_pascal_case(&variant.name) {
                        let suggested = to_pascal_case(&variant.name);
                        ctx.report_with_help(
                            self,
                            format!("variant `{}` should use PascalCase", variant.name),
                            Span::dummy(),
                            format!("rename to `{}`", suggested),
                        );
                    }
                }
            }
            Item::Trait(t) => {
                if !is_pascal_case(&t.name) {
                    let suggested = to_pascal_case(&t.name);
                    ctx.report_with_help(
                        self,
                        format!("trait `{}` should use PascalCase", t.name),
                        t.span,
                        format!("rename to `{}`", suggested),
                    );
                }
            }
            Item::TypeAlias(ta) => {
                if !is_pascal_case(&ta.name) {
                    let suggested = to_pascal_case(&ta.name);
                    ctx.report_with_help(
                        self,
                        format!("type alias `{}` should use PascalCase", ta.name),
                        ta.span,
                        format!("rename to `{}`", suggested),
                    );
                }
            }
            Item::Global(g) => {
                // Global constants should be SCREAMING_SNAKE_CASE
                if let Pattern::Binding { name, .. } = &g.pattern {
                    if !is_screaming_snake_case(name) {
                        let suggested = to_screaming_snake_case(name);
                        ctx.report_with_help(
                            self,
                            format!("global `{}` should use SCREAMING_SNAKE_CASE", name),
                            g.span,
                            format!("rename to `{}`", suggested),
                        );
                    }
                }
            }
            _ => {}
        }
    }

    fn check_stmt(&self, stmt: &Stmt, ctx: &mut LintContext) {
        // Check variable bindings in let statements
        if let Stmt::Let { pattern, .. } = stmt {
            if let Pattern::Binding { name, .. } = pattern {
                if !is_snake_case(name) && !name.starts_with('_') {
                    let suggested = to_snake_case(name);
                    ctx.report_with_help(
                        self,
                        format!("variable `{}` should use snake_case", name),
                        Span::dummy(),
                        format!("rename to `{}`", suggested),
                    );
                }
            }
        }
    }
}

/// Check if a name follows snake_case
fn is_snake_case(name: &str) -> bool {
    if name.is_empty() {
        return true;
    }

    // Allow underscore prefix for unused
    let name = name.trim_start_matches('_');
    if name.is_empty() {
        return true;
    }

    // Must start with lowercase
    if !name
        .chars()
        .next()
        .map_or(false, |c| c.is_lowercase() || c.is_numeric())
    {
        return false;
    }

    // Must only contain lowercase, digits, and underscores
    name.chars()
        .all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
}

/// Check if a name follows PascalCase
fn is_pascal_case(name: &str) -> bool {
    if name.is_empty() {
        return true;
    }

    // Must start with uppercase
    if !name.chars().next().map_or(false, |c| c.is_uppercase()) {
        return false;
    }

    // Must not contain underscores
    !name.contains('_')
}

/// Check if a name follows SCREAMING_SNAKE_CASE
fn is_screaming_snake_case(name: &str) -> bool {
    if name.is_empty() {
        return true;
    }

    // Must only contain uppercase, digits, and underscores
    name.chars()
        .all(|c| c.is_uppercase() || c.is_numeric() || c == '_')
}

/// Convert a name to snake_case
fn to_snake_case(name: &str) -> String {
    let mut result = String::new();
    let mut prev_lower = false;

    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if prev_lower && i > 0 {
                result.push('_');
            }
            result.push(c.to_ascii_lowercase());
            prev_lower = false;
        } else if c == '_' {
            result.push('_');
            prev_lower = false;
        } else {
            result.push(c);
            prev_lower = c.is_lowercase();
        }
    }

    result
}

/// Convert a name to PascalCase
fn to_pascal_case(name: &str) -> String {
    name.split('_')
        .filter(|s| !s.is_empty())
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(|c| c.to_lowercase()))
                    .collect(),
            }
        })
        .collect()
}

/// Convert a name to SCREAMING_SNAKE_CASE
fn to_screaming_snake_case(name: &str) -> String {
    to_snake_case(name).to_uppercase()
}

// =============================================================================
// Missing Documentation
// =============================================================================

/// Warns about public items without documentation
pub struct MissingDocumentation;

impl Lint for MissingDocumentation {
    fn id(&self) -> &'static str {
        "missing_docs"
    }

    fn name(&self) -> &'static str {
        "Missing Documentation"
    }

    fn description(&self) -> &'static str {
        "Warns about public items without documentation comments."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Documentation
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        // Note: Currently FnDef, StructDef, etc. don't have doc_comment fields.
        // This lint is a placeholder that can be activated when doc comments are added to the AST.
        match item {
            Item::Function(f) => {
                // Placeholder - would check f.doc_comment when available
                if !f.name.starts_with('_') {
                    // ctx.report_with_help(...)
                }
            }
            Item::Struct(s) => {
                if !s.name.starts_with('_') {
                    // ctx.report_with_help(...)
                }
            }
            Item::Enum(e) => {
                if !e.name.starts_with('_') {
                    // ctx.report_with_help(...)
                }
            }
            Item::Trait(t) => {
                if !t.name.starts_with('_') {
                    // ctx.report_with_help(...)
                }
            }
            _ => {}
        }
    }
}

// =============================================================================
// Redundant Semicolon
// =============================================================================

/// Detects redundant trailing semicolons
///
/// Trailing semicolons at the end of blocks change the return type to unit.
/// This lint warns when a semicolon might be redundant.
pub struct RedundantSemicolon;

impl Lint for RedundantSemicolon {
    fn id(&self) -> &'static str {
        "redundant_semicolon"
    }

    fn name(&self) -> &'static str {
        "Redundant Semicolon"
    }

    fn description(&self) -> &'static str {
        "Warns about redundant trailing semicolons in blocks."
    }

    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }

    fn category(&self) -> LintCategory {
        LintCategory::Style
    }

    fn check_stmt(&self, stmt: &Stmt, ctx: &mut LintContext) {
        // Check for expression statements followed by semicolons at end of block
        if let Stmt::Expr { expr, has_semi, .. } = stmt {
            // If this is an empty tuple literal with a semicolon, it's redundant
            if let Expr::Tuple { elements, .. } = expr {
                if elements.is_empty() && *has_semi {
                    ctx.report_with_suggestion(
                        self,
                        "redundant semicolon after unit expression",
                        Span::dummy(),
                        Suggestion {
                            message: "remove the semicolon".to_string(),
                            edits: vec![],
                            applicability: Applicability::MachineApplicable,
                        },
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_ids() {
        assert_eq!(UnusedVariable.id(), "unused_variable");
        assert_eq!(UnusedImport.id(), "unused_import");
        assert_eq!(NamingConvention.id(), "naming_convention");
        assert_eq!(RedundantSemicolon.id(), "redundant_semicolon");
    }

    #[test]
    fn test_lint_categories() {
        assert_eq!(UnusedVariable.category(), LintCategory::Style);
        assert_eq!(NamingConvention.category(), LintCategory::Naming);
    }

    #[test]
    fn test_is_snake_case() {
        assert!(is_snake_case("foo_bar"));
        assert!(is_snake_case("foo"));
        assert!(is_snake_case("foo_bar_baz"));
        assert!(is_snake_case("_foo"));
        assert!(is_snake_case("foo2"));
        assert!(is_snake_case("foo_2"));

        assert!(!is_snake_case("FooBar"));
        assert!(!is_snake_case("fooBar"));
        assert!(!is_snake_case("FOO_BAR"));
    }

    #[test]
    fn test_is_pascal_case() {
        assert!(is_pascal_case("FooBar"));
        assert!(is_pascal_case("Foo"));
        assert!(is_pascal_case("FooBarBaz"));

        assert!(!is_pascal_case("foo_bar"));
        assert!(!is_pascal_case("fooBar"));
        assert!(!is_pascal_case("Foo_Bar"));
    }

    #[test]
    fn test_is_screaming_snake_case() {
        assert!(is_screaming_snake_case("FOO_BAR"));
        assert!(is_screaming_snake_case("FOO"));
        assert!(is_screaming_snake_case("FOO_BAR_BAZ"));
        assert!(is_screaming_snake_case("FOO2"));

        assert!(!is_screaming_snake_case("foo_bar"));
        assert!(!is_screaming_snake_case("FooBar"));
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("FooBar"), "foo_bar");
        assert_eq!(to_snake_case("fooBar"), "foo_bar");
        assert_eq!(to_snake_case("FOO_BAR"), "foo_bar"); // Already snake_case, just uppercase
        assert_eq!(to_snake_case("foo_bar"), "foo_bar");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("foo_bar"), "FooBar");
        assert_eq!(to_pascal_case("foo"), "Foo");
        assert_eq!(to_pascal_case("foo_bar_baz"), "FooBarBaz");
    }

    #[test]
    fn test_to_screaming_snake_case() {
        assert_eq!(to_screaming_snake_case("fooBar"), "FOO_BAR");
        assert_eq!(to_screaming_snake_case("FooBar"), "FOO_BAR");
    }
}
