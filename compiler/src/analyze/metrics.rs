//! Code metrics calculation
//!
//! Calculates various code metrics including:
//! - Lines of code
//! - Cyclomatic complexity
//! - Cognitive complexity
//! - Function metrics

use crate::ast::{Ast, Block, Expr, FnDef, Item, Stmt};
use std::collections::HashMap;

/// Code metrics for a file
#[derive(Debug, Default, Clone)]
pub struct FileMetrics {
    /// Total lines of code
    pub loc: usize,

    /// Lines of comments
    pub comment_lines: usize,

    /// Blank lines
    pub blank_lines: usize,

    /// Number of functions
    pub functions: usize,

    /// Number of types (struct + enum)
    pub types: usize,

    /// Number of traits
    pub traits: usize,

    /// Number of impl blocks
    pub impls: usize,

    /// Average function length
    pub avg_function_length: f64,

    /// Maximum function length
    pub max_function_length: usize,

    /// Average cyclomatic complexity
    pub avg_complexity: f64,

    /// Maximum cyclomatic complexity
    pub max_complexity: usize,

    /// Per-function metrics
    pub function_metrics: HashMap<String, FunctionMetrics>,

    /// Total statements
    pub total_statements: usize,

    /// Total expressions
    pub total_expressions: usize,
}

/// Metrics for a single function
#[derive(Debug, Default, Clone)]
pub struct FunctionMetrics {
    /// Function name
    pub name: String,

    /// Number of lines (approximate)
    pub lines: usize,

    /// Cyclomatic complexity
    pub cyclomatic_complexity: usize,

    /// Cognitive complexity
    pub cognitive_complexity: usize,

    /// Number of parameters
    pub parameters: usize,

    /// Number of return statements
    pub returns: usize,

    /// Maximum nesting depth
    pub nesting_depth: usize,

    /// Number of local variables
    pub local_variables: usize,

    /// Number of loops
    pub loops: usize,

    /// Number of branches (if/match)
    pub branches: usize,

    /// Called functions
    pub calls: Vec<String>,

    /// Declared effects
    pub effects: Vec<String>,
}

/// Calculate metrics for an AST
pub fn calculate_metrics(ast: &Ast, source: &str) -> FileMetrics {
    let mut metrics = FileMetrics::default();

    // Count lines
    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            metrics.blank_lines += 1;
        } else if trimmed.starts_with("//") || trimmed.starts_with("/*") {
            metrics.comment_lines += 1;
        } else {
            metrics.loc += 1;
        }
    }

    // Analyze items
    for item in &ast.items {
        analyze_item(item, &mut metrics);
    }

    // Calculate averages
    if metrics.functions > 0 {
        let total_length: usize = metrics.function_metrics.values().map(|m| m.lines).sum();
        metrics.avg_function_length = total_length as f64 / metrics.functions as f64;

        metrics.max_function_length = metrics
            .function_metrics
            .values()
            .map(|m| m.lines)
            .max()
            .unwrap_or(0);

        let total_complexity: usize = metrics
            .function_metrics
            .values()
            .map(|m| m.cyclomatic_complexity)
            .sum();
        metrics.avg_complexity = total_complexity as f64 / metrics.functions as f64;

        metrics.max_complexity = metrics
            .function_metrics
            .values()
            .map(|m| m.cyclomatic_complexity)
            .max()
            .unwrap_or(0);
    }

    metrics
}

fn analyze_item(item: &Item, metrics: &mut FileMetrics) {
    match item {
        Item::Function(f) => {
            metrics.functions += 1;
            let func_metrics = analyze_function(f);
            metrics
                .function_metrics
                .insert(f.name.clone(), func_metrics);
        }
        Item::Struct(_) => {
            metrics.types += 1;
        }
        Item::Enum(_) => {
            metrics.types += 1;
        }
        Item::Trait(t) => {
            metrics.traits += 1;
            // Trait methods are TraitItem, not Item - skip for now
            let _ = t;
        }
        Item::Impl(i) => {
            metrics.impls += 1;
            // Impl items are ImplItem, not Item - skip for now
            let _ = i;
        }
        _ => {}
    }
}

fn analyze_function(func: &FnDef) -> FunctionMetrics {
    let effects: Vec<String> = func
        .effects
        .iter()
        .map(|e| e.name.segments.join("::"))
        .collect();

    let mut metrics = FunctionMetrics {
        name: func.name.clone(),
        parameters: func.params.len(),
        effects,
        ..Default::default()
    };

    // FnDef.body is Block, not Option<Block>
    let body = &func.body;
    metrics.lines = count_block_lines(body);
    metrics.cyclomatic_complexity = cyclomatic_complexity(body);
    metrics.cognitive_complexity = cognitive_complexity(body, 0);
    metrics.nesting_depth = max_nesting_depth(body, 0);

    // Count various elements
    let counts = count_elements(body);
    metrics.local_variables = counts.variables;
    metrics.loops = counts.loops;
    metrics.branches = counts.branches;
    metrics.returns = counts.returns;
    metrics.calls = counts.calls;

    metrics
}

fn count_block_lines(block: &Block) -> usize {
    let mut count = 0;

    for stmt in &block.stmts {
        count += count_stmt_lines(stmt);
    }

    count.max(1) // At least 1 line for {}
}

fn count_stmt_lines(stmt: &Stmt) -> usize {
    match stmt {
        Stmt::Let { value, .. } => 1 + value.as_ref().map(count_expr_lines).unwrap_or(0),
        Stmt::Expr { expr, .. } => count_expr_lines(expr) + 1,
        Stmt::Assign { target, value, .. } => {
            count_expr_lines(target) + count_expr_lines(value) + 1
        }
        Stmt::Empty | Stmt::MacroInvocation(_) => 1,
    }
}

fn count_expr_lines(expr: &Expr) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            1 + count_block_lines(then_branch)
                + else_branch
                    .as_ref()
                    .map(|e| count_expr_lines(e))
                    .unwrap_or(0)
        }
        Expr::Match { arms, .. } => {
            1 + arms.len()
                + arms
                    .iter()
                    .map(|a| count_expr_lines(&a.body))
                    .sum::<usize>()
        }
        Expr::Block { block, .. } => count_block_lines(block),
        Expr::Loop { body, .. } => 1 + count_block_lines(body),
        Expr::While { body, .. } => 1 + count_block_lines(body),
        Expr::For { body, .. } => 1 + count_block_lines(body),
        Expr::Closure { body, .. } => 1 + count_expr_lines(body),
        _ => 1,
    }
}

/// Calculate cyclomatic complexity of a block
pub fn cyclomatic_complexity(block: &Block) -> usize {
    let mut complexity = 1; // Base complexity

    for stmt in &block.stmts {
        complexity += count_decision_points_stmt(stmt);
    }

    complexity
}

fn count_decision_points_stmt(stmt: &Stmt) -> usize {
    match stmt {
        Stmt::Expr { expr, .. } => count_decision_points_expr(expr),
        Stmt::Let { value, .. } => value.as_ref().map(count_decision_points_expr).unwrap_or(0),
        Stmt::Assign { target, value, .. } => {
            count_decision_points_expr(target) + count_decision_points_expr(value)
        }
        Stmt::Empty | Stmt::MacroInvocation(_) => 0,
    }
}

fn count_decision_points_expr(expr: &Expr) -> usize {
    match expr {
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            1 + count_decision_points_expr(condition)
                + count_decision_points_block(then_branch)
                + else_branch
                    .as_ref()
                    .map(|e| count_decision_points_expr(e))
                    .unwrap_or(0)
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            count_decision_points_expr(scrutinee)
                + arms.len().saturating_sub(1)
                + arms
                    .iter()
                    .map(|a| {
                        count_decision_points_expr(&a.body)
                            + a.guard
                                .as_ref()
                                .map(|g| 1 + count_decision_points_expr(g))
                                .unwrap_or(0)
                    })
                    .sum::<usize>()
        }
        Expr::While {
            condition, body, ..
        } => 1 + count_decision_points_expr(condition) + count_decision_points_block(body),
        Expr::For { iter, body, .. } => {
            1 + count_decision_points_expr(iter) + count_decision_points_block(body)
        }
        Expr::Loop { body, .. } => count_decision_points_block(body),
        Expr::Binary {
            op, left, right, ..
        } => {
            // Check for logical operators (And/Or add decision points)
            let op_points = match op {
                crate::ast::BinaryOp::And | crate::ast::BinaryOp::Or => 1,
                _ => 0,
            };
            op_points + count_decision_points_expr(left) + count_decision_points_expr(right)
        }
        Expr::Block { block, .. } => count_decision_points_block(block),
        Expr::Try { expr, .. } => 1 + count_decision_points_expr(expr),
        _ => 0,
    }
}

fn count_decision_points_block(block: &Block) -> usize {
    let mut count = 0;
    for stmt in &block.stmts {
        count += count_decision_points_stmt(stmt);
    }
    count
}

/// Calculate cognitive complexity of a block
pub fn cognitive_complexity(block: &Block, nesting: usize) -> usize {
    let mut complexity = 0;

    for stmt in &block.stmts {
        complexity += cognitive_complexity_stmt(stmt, nesting);
    }

    complexity
}

fn cognitive_complexity_stmt(stmt: &Stmt, nesting: usize) -> usize {
    match stmt {
        Stmt::Expr { expr, .. } => cognitive_complexity_expr(expr, nesting),
        Stmt::Let { value, .. } => value
            .as_ref()
            .map(|e| cognitive_complexity_expr(e, nesting))
            .unwrap_or(0),
        Stmt::Assign { target, value, .. } => {
            cognitive_complexity_expr(target, nesting) + cognitive_complexity_expr(value, nesting)
        }
        Stmt::Empty | Stmt::MacroInvocation(_) => 0,
    }
}

fn cognitive_complexity_expr(expr: &Expr, nesting: usize) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            // +1 for if, + nesting increment
            let base = 1 + nesting;
            let then_cost = cognitive_complexity(then_branch, nesting + 1);
            let else_cost = else_branch
                .as_ref()
                .map(|e| 1 + cognitive_complexity_expr(e, nesting + 1))
                .unwrap_or(0);
            base + then_cost + else_cost
        }
        Expr::While { body, .. } | Expr::For { body, .. } => {
            1 + nesting + cognitive_complexity(body, nesting + 1)
        }
        Expr::Loop { body, .. } => 1 + nesting + cognitive_complexity(body, nesting + 1),
        Expr::Match { arms, .. } => {
            // +1 for match
            1 + arms.len().saturating_sub(1)
                + arms
                    .iter()
                    .map(|a| cognitive_complexity_expr(&a.body, nesting + 1))
                    .sum::<usize>()
        }
        Expr::Binary {
            op, left, right, ..
        } => {
            let is_logical = matches!(op, crate::ast::BinaryOp::And | crate::ast::BinaryOp::Or);
            let base = if is_logical { 1 } else { 0 };
            base + cognitive_complexity_expr(left, nesting)
                + cognitive_complexity_expr(right, nesting)
        }
        Expr::Block { block, .. } => cognitive_complexity(block, nesting),
        Expr::Break { .. } | Expr::Continue { .. } => 1,
        Expr::Try { expr, .. } => 1 + cognitive_complexity_expr(expr, nesting),
        _ => 0,
    }
}

/// Calculate maximum nesting depth
pub fn max_nesting_depth(block: &Block, current: usize) -> usize {
    let mut max = current;

    for stmt in &block.stmts {
        max = max.max(nesting_depth_stmt(stmt, current));
    }

    max
}

fn nesting_depth_stmt(stmt: &Stmt, current: usize) -> usize {
    match stmt {
        Stmt::Expr { expr, .. } => nesting_depth_expr(expr, current),
        Stmt::Assign { target, value, .. } => {
            nesting_depth_expr(target, current).max(nesting_depth_expr(value, current))
        }
        Stmt::Let { .. } | Stmt::Empty | Stmt::MacroInvocation(_) => current,
    }
}

fn nesting_depth_expr(expr: &Expr, current: usize) -> usize {
    match expr {
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            let then_depth = max_nesting_depth(then_branch, current + 1);
            let else_depth = else_branch
                .as_ref()
                .map(|e| nesting_depth_expr(e, current + 1))
                .unwrap_or(current);
            then_depth.max(else_depth)
        }
        Expr::Block { block, .. } => max_nesting_depth(block, current + 1),
        Expr::Match { arms, .. } => arms
            .iter()
            .map(|a| nesting_depth_expr(&a.body, current + 1))
            .max()
            .unwrap_or(current),
        Expr::Loop { body, .. } | Expr::While { body, .. } | Expr::For { body, .. } => {
            max_nesting_depth(body, current + 1)
        }
        _ => current,
    }
}

/// Element counts
struct ElementCounts {
    variables: usize,
    loops: usize,
    branches: usize,
    returns: usize,
    calls: Vec<String>,
}

fn count_elements(block: &Block) -> ElementCounts {
    let mut counts = ElementCounts {
        variables: 0,
        loops: 0,
        branches: 0,
        returns: 0,
        calls: Vec::new(),
    };

    for stmt in &block.stmts {
        count_elements_stmt(stmt, &mut counts);
    }

    counts
}

fn count_elements_stmt(stmt: &Stmt, counts: &mut ElementCounts) {
    match stmt {
        Stmt::Let { value, .. } => {
            counts.variables += 1;
            if let Some(e) = value {
                count_elements_expr(e, counts);
            }
        }
        Stmt::Expr { expr, .. } => count_elements_expr(expr, counts),
        Stmt::Assign { target, value, .. } => {
            count_elements_expr(target, counts);
            count_elements_expr(value, counts);
        }
        Stmt::Empty | Stmt::MacroInvocation(_) => {}
    }
}

fn count_elements_expr(expr: &Expr, counts: &mut ElementCounts) {
    match expr {
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            counts.branches += 1;
            count_elements_expr(condition, counts);
            count_elements_block(then_branch, counts);
            if let Some(e) = else_branch {
                count_elements_expr(e, counts);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            counts.branches += arms.len();
            count_elements_expr(scrutinee, counts);
            for arm in arms {
                count_elements_expr(&arm.body, counts);
            }
        }
        Expr::Loop { body, .. } => {
            counts.loops += 1;
            count_elements_block(body, counts);
        }
        Expr::While {
            condition, body, ..
        } => {
            counts.loops += 1;
            count_elements_expr(condition, counts);
            count_elements_block(body, counts);
        }
        Expr::For { iter, body, .. } => {
            counts.loops += 1;
            count_elements_expr(iter, counts);
            count_elements_block(body, counts);
        }
        Expr::Return { value, .. } => {
            counts.returns += 1;
            if let Some(v) = value {
                count_elements_expr(v, counts);
            }
        }
        Expr::Call { callee, args, .. } => {
            if let Expr::Path { path, .. } = callee.as_ref() {
                counts.calls.push(path.segments.join("::"));
            }
            count_elements_expr(callee, counts);
            for arg in args {
                count_elements_expr(arg, counts);
            }
        }
        Expr::MethodCall {
            receiver,
            method,
            args,
            ..
        } => {
            counts.calls.push(format!("_.{}", method));
            count_elements_expr(receiver, counts);
            for arg in args {
                count_elements_expr(arg, counts);
            }
        }
        Expr::Block { block, .. } => count_elements_block(block, counts),
        Expr::Binary { left, right, .. } => {
            count_elements_expr(left, counts);
            count_elements_expr(right, counts);
        }
        Expr::Unary { expr, .. } => count_elements_expr(expr, counts),
        _ => {}
    }
}

fn count_elements_block(block: &Block, counts: &mut ElementCounts) {
    for stmt in &block.stmts {
        count_elements_stmt(stmt, counts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complexity_calculation() {
        // Complexity should be 1 for empty block
        let block = Block { stmts: Vec::new() };
        assert_eq!(cyclomatic_complexity(&block), 1);
    }
}
