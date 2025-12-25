//! Built-in lint rules
//!
//! Organized by category:
//! - Correctness: Likely bugs
//! - Style: Code style issues
//! - Complexity: Overly complex code
//! - Performance: Performance issues
//! - Effects: Effect system issues
//! - Safety: Safety concerns

use super::*;
use crate::ast::*;

// =============================================================================
// Correctness Lints
// =============================================================================

/// Unused variable lint
pub struct UnusedVariable;

impl Lint for UnusedVariable {
    fn id(&self) -> &'static str {
        "unused_variable"
    }
    fn name(&self) -> &'static str {
        "Unused Variable"
    }
    fn description(&self) -> &'static str {
        "Detects variables that are declared but never used"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn fix(&self, diagnostic: &LintDiagnostic) -> Option<Fix> {
        Some(Fix {
            message: "prefix with underscore to indicate intentionally unused".into(),
            edits: vec![Edit {
                span: Span {
                    start: diagnostic.span.start,
                    end: diagnostic.span.start,
                },
                replacement: "_".into(),
            }],
        })
    }
}

/// Unused import lint
pub struct UnusedImport;

impl Lint for UnusedImport {
    fn id(&self) -> &'static str {
        "unused_import"
    }
    fn name(&self) -> &'static str {
        "Unused Import"
    }
    fn description(&self) -> &'static str {
        "Detects imports that are not used"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn check_crate(&self, _ast: &Ast, _ctx: &mut LintContext) {
        // Would analyze imports vs usage across the crate
    }

    fn fix(&self, diagnostic: &LintDiagnostic) -> Option<Fix> {
        Some(Fix {
            message: "remove unused import".into(),
            edits: vec![Edit {
                span: diagnostic.span,
                replacement: "".into(),
            }],
        })
    }
}

/// Unused function lint
pub struct UnusedFunction;

impl Lint for UnusedFunction {
    fn id(&self) -> &'static str {
        "unused_function"
    }
    fn name(&self) -> &'static str {
        "Unused Function"
    }
    fn description(&self) -> &'static str {
        "Detects private functions that are never called"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            // Skip public functions, main, and test functions
            if matches!(f.visibility, Visibility::Public)
                || f.name == "main"
                || f.name.starts_with("test_")
            {}

            // Would check if function is called anywhere
            // For now, just register it for dead code analysis
        }
    }
}

/// Unreachable code lint
pub struct UnreachableCode;

impl Lint for UnreachableCode {
    fn id(&self) -> &'static str {
        "unreachable_code"
    }
    fn name(&self) -> &'static str {
        "Unreachable Code"
    }
    fn description(&self) -> &'static str {
        "Detects code that can never be executed"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn check_stmt(&self, _stmt: &Stmt, _ctx: &mut LintContext) {
        // Would check for statements after return/break/continue
    }
}

/// Division by zero lint
pub struct DivisionByZero;

impl Lint for DivisionByZero {
    fn id(&self) -> &'static str {
        "division_by_zero"
    }
    fn name(&self) -> &'static str {
        "Division by Zero"
    }
    fn description(&self) -> &'static str {
        "Detects divisions where the divisor is known to be zero"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Deny
    }
    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        if let Expr::Binary { op, right, .. } = expr
            && matches!(op, BinaryOp::Div | BinaryOp::Rem)
            && is_zero(right)
        {
            ctx.report(self, "division by zero", Span::dummy());
        }
    }
}

fn is_zero(expr: &Expr) -> bool {
    match expr {
        Expr::Literal {
            value: Literal::Int(0),
            ..
        } => true,
        Expr::Literal {
            value: Literal::Float(f),
            ..
        } if *f == 0.0 => true,
        _ => false,
    }
}

/// Infinite loop lint
pub struct InfiniteLoop;

impl Lint for InfiniteLoop {
    fn id(&self) -> &'static str {
        "infinite_loop"
    }
    fn name(&self) -> &'static str {
        "Infinite Loop"
    }
    fn description(&self) -> &'static str {
        "Detects loops that appear to never terminate"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Correctness
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        if let Expr::While {
            condition, body, ..
        } = expr
        {
            // Check for `while true` without break
            if is_always_true(condition) && !contains_break(body) {
                ctx.report_with_help(
                    self,
                    "this loop appears to never terminate",
                    Span::dummy(),
                    "add a `break` condition or use `loop { ... }` for intentional infinite loops",
                );
            }
        }
    }
}

fn is_always_true(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Literal {
            value: Literal::Bool(true),
            ..
        }
    )
}

fn contains_break(block: &Block) -> bool {
    fn check_stmt(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr { expr, .. } => check_expr(expr),
            _ => false,
        }
    }

    fn check_expr(expr: &Expr) -> bool {
        match expr {
            Expr::Break { .. } => true,
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                contains_break(then_branch)
                    || else_branch.as_ref().map(|e| check_expr(e)).unwrap_or(false)
            }
            Expr::Block { block, .. } => contains_break(block),
            Expr::Match { arms, .. } => arms.iter().any(|a| check_expr(&a.body)),
            _ => false,
        }
    }

    block.stmts.iter().any(check_stmt)
}

// =============================================================================
// Style Lints
// =============================================================================

/// Naming convention lint
pub struct NamingConvention;

impl Lint for NamingConvention {
    fn id(&self) -> &'static str {
        "naming_convention"
    }
    fn name(&self) -> &'static str {
        "Naming Convention"
    }
    fn description(&self) -> &'static str {
        "Enforces D naming conventions"
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
                if !is_snake_case(&f.name) && !f.name.starts_with('_') {
                    ctx.report_with_help(
                        self,
                        format!("function `{}` should be snake_case", f.name),
                        f.span,
                        format!("rename to `{}`", to_snake_case(&f.name)),
                    );
                }
            }
            Item::Struct(s) => {
                // Structs should be PascalCase
                if !is_pascal_case(&s.name) {
                    ctx.report_with_help(
                        self,
                        format!("struct `{}` should be PascalCase", s.name),
                        s.span,
                        format!("rename to `{}`", to_pascal_case(&s.name)),
                    );
                }
            }
            Item::Enum(e) => {
                // Enums should be PascalCase
                if !is_pascal_case(&e.name) {
                    ctx.report_with_help(
                        self,
                        format!("enum `{}` should be PascalCase", e.name),
                        e.span,
                        format!("rename to `{}`", to_pascal_case(&e.name)),
                    );
                }
            }
            Item::Global(g) if g.is_const => {
                // Constants should be SCREAMING_SNAKE_CASE
                if let Pattern::Binding { name, .. } = &g.pattern
                    && !is_screaming_snake_case(name)
                {
                    ctx.report_with_help(
                        self,
                        format!("constant `{}` should be SCREAMING_SNAKE_CASE", name),
                        g.span,
                        format!("rename to `{}`", to_screaming_snake_case(name)),
                    );
                }
            }
            Item::Trait(t) => {
                // Traits should be PascalCase
                if !is_pascal_case(&t.name) {
                    ctx.report_with_help(
                        self,
                        format!("trait `{}` should be PascalCase", t.name),
                        t.span,
                        format!("rename to `{}`", to_pascal_case(&t.name)),
                    );
                }
            }
            _ => {}
        }
    }
}

fn is_snake_case(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    s.chars()
        .all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
}

fn is_pascal_case(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) && !s.contains('_')
}

fn is_screaming_snake_case(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    s.chars()
        .all(|c| c.is_uppercase() || c.is_numeric() || c == '_')
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut chars = word.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().chain(chars).collect(),
            }
        })
        .collect()
}

fn to_screaming_snake_case(s: &str) -> String {
    to_snake_case(s).to_uppercase()
}

/// Missing documentation lint
pub struct MissingDocumentation;

impl Lint for MissingDocumentation {
    fn id(&self) -> &'static str {
        "missing_docs"
    }
    fn name(&self) -> &'static str {
        "Missing Documentation"
    }
    fn description(&self) -> &'static str {
        "Detects public items without documentation"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Allow
    }
    fn category(&self) -> LintCategory {
        LintCategory::Documentation
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        match item {
            Item::Function(f) => {
                if matches!(f.visibility, Visibility::Public)
                    && !has_doc_comment(&f.span, ctx.source)
                {
                    ctx.report_with_help(
                        self,
                        format!("public function `{}` is missing documentation", f.name),
                        f.span,
                        "add a doc comment with `///`",
                    );
                }
            }
            Item::Struct(s) => {
                if matches!(s.visibility, Visibility::Public)
                    && !has_doc_comment(&s.span, ctx.source)
                {
                    ctx.report_with_help(
                        self,
                        format!("public struct `{}` is missing documentation", s.name),
                        s.span,
                        "add a doc comment with `///`",
                    );
                }
            }
            Item::Enum(e) => {
                if matches!(e.visibility, Visibility::Public)
                    && !has_doc_comment(&e.span, ctx.source)
                {
                    ctx.report_with_help(
                        self,
                        format!("public enum `{}` is missing documentation", e.name),
                        e.span,
                        "add a doc comment with `///`",
                    );
                }
            }
            Item::Trait(t) => {
                if matches!(t.visibility, Visibility::Public)
                    && !has_doc_comment(&t.span, ctx.source)
                {
                    ctx.report_with_help(
                        self,
                        format!("public trait `{}` is missing documentation", t.name),
                        t.span,
                        "add a doc comment with `///`",
                    );
                }
            }
            _ => {}
        }
    }
}

fn has_doc_comment(span: &Span, source: &str) -> bool {
    // Look backwards from span.start for /// or /**
    if span.start == 0 {
        return false;
    }

    let before = &source[..span.start];
    let lines: Vec<&str> = before.lines().collect();

    if let Some(last_line) = lines.last() {
        let trimmed = last_line.trim();
        trimmed.starts_with("///") || trimmed.starts_with("/**")
    } else {
        false
    }
}

// =============================================================================
// Complexity Lints
// =============================================================================

/// Too many arguments lint
pub struct TooManyArguments;

impl Lint for TooManyArguments {
    fn id(&self) -> &'static str {
        "too_many_arguments"
    }
    fn name(&self) -> &'static str {
        "Too Many Arguments"
    }
    fn description(&self) -> &'static str {
        "Detects functions with too many parameters"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            const MAX_ARGS: usize = 7;

            if f.params.len() > MAX_ARGS {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` has {} parameters (max {})",
                        f.name,
                        f.params.len(),
                        MAX_ARGS
                    ),
                    f.span,
                    "consider grouping parameters into a struct",
                );
            }
        }
    }
}

/// Too long function lint
pub struct TooLongFunction;

impl Lint for TooLongFunction {
    fn id(&self) -> &'static str {
        "too_long_function"
    }
    fn name(&self) -> &'static str {
        "Too Long Function"
    }
    fn description(&self) -> &'static str {
        "Detects functions that are too long"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            const MAX_LINES: usize = 100;

            let line_count = count_lines(&f.body);
            if line_count > MAX_LINES {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` is {} lines long (max {})",
                        f.name, line_count, MAX_LINES
                    ),
                    f.span,
                    "consider breaking this function into smaller functions",
                );
            }
        }
    }
}

fn count_lines(block: &Block) -> usize {
    let mut count = block.stmts.len();

    for stmt in &block.stmts {
        if let Stmt::Expr { expr, .. } = stmt {
            count += count_expr_lines(expr);
        }
    }

    count
}

fn count_expr_lines(expr: &Expr) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let mut count = count_lines(then_branch);
            if let Some(e) = else_branch {
                count += count_expr_lines(e);
            }
            count
        }
        Expr::Block { block, .. } => count_lines(block),
        Expr::Match { arms, .. } => arms.len(),
        Expr::Loop { body, .. } => count_lines(body),
        Expr::While { body, .. } => count_lines(body),
        Expr::For { body, .. } => count_lines(body),
        _ => 0,
    }
}

/// Deep nesting lint
pub struct DeepNesting;

impl Lint for DeepNesting {
    fn id(&self) -> &'static str {
        "deep_nesting"
    }
    fn name(&self) -> &'static str {
        "Deep Nesting"
    }
    fn description(&self) -> &'static str {
        "Detects deeply nested code blocks"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            const MAX_NESTING: usize = 5;

            let max_depth = max_nesting_depth(&f.body, 0);
            if max_depth > MAX_NESTING {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` has nesting depth of {} (max {})",
                        f.name, max_depth, MAX_NESTING
                    ),
                    f.span,
                    "consider extracting nested code into separate functions",
                );
            }
        }
    }
}

fn max_nesting_depth(block: &Block, current: usize) -> usize {
    let mut max = current;

    for stmt in &block.stmts {
        if let Stmt::Expr { expr, .. } = stmt {
            max = max.max(expr_nesting_depth(expr, current + 1));
        }
    }

    max
}

fn expr_nesting_depth(expr: &Expr, current: usize) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let mut max = max_nesting_depth(then_branch, current);
            if let Some(e) = else_branch {
                max = max.max(expr_nesting_depth(e, current));
            }
            max
        }
        Expr::Block { block, .. } => max_nesting_depth(block, current),
        Expr::Match { arms, .. } => arms
            .iter()
            .map(|a| expr_nesting_depth(&a.body, current + 1))
            .max()
            .unwrap_or(current),
        Expr::Loop { body, .. } => max_nesting_depth(body, current),
        Expr::While { body, .. } => max_nesting_depth(body, current),
        Expr::For { body, .. } => max_nesting_depth(body, current),
        _ => current,
    }
}

/// Cyclomatic complexity lint
pub struct CyclomaticComplexity;

impl Lint for CyclomaticComplexity {
    fn id(&self) -> &'static str {
        "cyclomatic_complexity"
    }
    fn name(&self) -> &'static str {
        "Cyclomatic Complexity"
    }
    fn description(&self) -> &'static str {
        "Detects functions with high cyclomatic complexity"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Complexity
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item {
            const MAX_COMPLEXITY: usize = 15;

            let complexity = calculate_cyclomatic_complexity(&f.body);
            if complexity > MAX_COMPLEXITY {
                ctx.report_with_help(
                    self,
                    format!(
                        "function `{}` has cyclomatic complexity of {} (max {})",
                        f.name, complexity, MAX_COMPLEXITY
                    ),
                    f.span,
                    "consider breaking this function into smaller pieces",
                );
            }
        }
    }
}

fn calculate_cyclomatic_complexity(block: &Block) -> usize {
    let mut complexity = 1; // Base complexity

    for stmt in &block.stmts {
        complexity += count_decision_points_stmt(stmt);
    }

    complexity
}

fn count_decision_points_stmt(stmt: &Stmt) -> usize {
    match stmt {
        Stmt::Expr { expr, .. } => count_decision_points_expr(expr),
        _ => 0,
    }
}

fn count_decision_points_expr(expr: &Expr) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            1 + calculate_cyclomatic_complexity(then_branch)
                - 1 // Don't double count base
                + else_branch
                    .as_ref()
                    .map(|e| count_decision_points_expr(e))
                    .unwrap_or(0)
        }
        Expr::Match { arms, .. } => {
            arms.len().saturating_sub(1)
                + arms
                    .iter()
                    .map(|a| count_decision_points_expr(&a.body))
                    .sum::<usize>()
        }
        Expr::While { body, .. } => 1 + calculate_cyclomatic_complexity(body) - 1,
        Expr::For { body, .. } => 1 + calculate_cyclomatic_complexity(body) - 1,
        Expr::Loop { body, .. } => calculate_cyclomatic_complexity(body) - 1,
        Expr::Binary {
            op, left, right, ..
        } if matches!(op, BinaryOp::And | BinaryOp::Or) => {
            1 + count_decision_points_expr(left) + count_decision_points_expr(right)
        }
        Expr::Block { block, .. } => calculate_cyclomatic_complexity(block) - 1,
        _ => 0,
    }
}

// =============================================================================
// Performance Lints
// =============================================================================

/// Unnecessary clone lint
pub struct UnnecessaryClone;

impl Lint for UnnecessaryClone {
    fn id(&self) -> &'static str {
        "unnecessary_clone"
    }
    fn name(&self) -> &'static str {
        "Unnecessary Clone"
    }
    fn description(&self) -> &'static str {
        "Detects clones that might not be needed"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        if let Expr::MethodCall {
            method, receiver, ..
        } = expr
            && method == "clone"
        {
            // Check if clone result is immediately passed to a function
            // This is a simplified check - a full check would need type info
            ctx.report_with_help(
                self,
                "consider whether this clone is necessary",
                Span::dummy(),
                "cloning has a runtime cost; use references when possible",
            );
        }
    }
}

/// Large stack value lint
pub struct LargeStackValue;

impl Lint for LargeStackValue {
    fn id(&self) -> &'static str {
        "large_stack_value"
    }
    fn name(&self) -> &'static str {
        "Large Stack Value"
    }
    fn description(&self) -> &'static str {
        "Detects large values that might overflow the stack"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Performance
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Check for large array literals
        if let Expr::Array { elements, .. } = expr {
            const MAX_STACK_ARRAY: usize = 1000;

            if elements.len() > MAX_STACK_ARRAY {
                ctx.report_with_help(
                    self,
                    format!(
                        "array literal with {} elements might overflow the stack",
                        elements.len()
                    ),
                    Span::dummy(),
                    "consider using Vec for large collections",
                );
            }
        }
    }
}

// =============================================================================
// Effect Lints
// =============================================================================

/// Unused effect lint
pub struct UnusedEffect;

impl Lint for UnusedEffect {
    fn id(&self) -> &'static str {
        "unused_effect"
    }
    fn name(&self) -> &'static str {
        "Unused Effect"
    }
    fn description(&self) -> &'static str {
        "Detects effect annotations that are not used in the function body"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Effects
    }

    fn check_item(&self, item: &Item, ctx: &mut LintContext) {
        if let Item::Function(f) = item
            && !f.effects.is_empty()
        {
            // Would analyze if declared effects are actually performed
            // For now, just skip
        }
    }
}

/// Missing effect annotation lint
pub struct MissingEffectAnnotation;

impl Lint for MissingEffectAnnotation {
    fn id(&self) -> &'static str {
        "missing_effect"
    }
    fn name(&self) -> &'static str {
        "Missing Effect Annotation"
    }
    fn description(&self) -> &'static str {
        "Detects functions that perform effects without declaring them"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Warn
    }
    fn category(&self) -> LintCategory {
        LintCategory::Effects
    }

    fn check_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        if let Expr::Perform { effect, .. } = expr {
            // Would check if effect is declared in current function
            // For now, just skip
        }
    }
}

// =============================================================================
// Safety Lints
// =============================================================================

/// Unsafe block lint
pub struct UnsafeBlock;

impl Lint for UnsafeBlock {
    fn id(&self) -> &'static str {
        "unsafe_block"
    }
    fn name(&self) -> &'static str {
        "Unsafe Block"
    }
    fn description(&self) -> &'static str {
        "Warns about usage of unsafe blocks"
    }
    fn default_level(&self) -> LintLevel {
        LintLevel::Allow
    }
    fn category(&self) -> LintCategory {
        LintCategory::Safety
    }

    fn check_expr(&self, _expr: &Expr, _ctx: &mut LintContext) {
        // Would check for unsafe blocks when we add them to the language
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_snake_case() {
        assert!(is_snake_case("hello_world"));
        assert!(is_snake_case("foo"));
        assert!(is_snake_case("foo_bar_baz"));
        assert!(!is_snake_case("HelloWorld"));
        assert!(!is_snake_case("helloWorld"));
    }

    #[test]
    fn test_is_pascal_case() {
        assert!(is_pascal_case("HelloWorld"));
        assert!(is_pascal_case("Foo"));
        assert!(!is_pascal_case("hello_world"));
        assert!(!is_pascal_case("helloWorld"));
    }

    #[test]
    fn test_to_snake_case() {
        assert_eq!(to_snake_case("HelloWorld"), "hello_world");
        assert_eq!(to_snake_case("Foo"), "foo");
        assert_eq!(to_snake_case("FooBarBaz"), "foo_bar_baz");
    }

    #[test]
    fn test_to_pascal_case() {
        assert_eq!(to_pascal_case("hello_world"), "HelloWorld");
        assert_eq!(to_pascal_case("foo"), "Foo");
        assert_eq!(to_pascal_case("foo_bar_baz"), "FooBarBaz");
    }
}
