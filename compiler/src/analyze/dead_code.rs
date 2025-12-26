//! Dead Code Detection
//!
//! Analyzes code to find:
//! - Unused functions, types, and constants
//! - Unreachable code paths
//! - Unused imports and variables
//! - Shadowed definitions

use crate::ast::{Ast, Block, Expr, Item, Pattern, Stmt, TypeExpr, Visibility};
use crate::common::Span;
use std::collections::{HashMap, HashSet};

/// Extract the primary name from a pattern (for display/tracking purposes)
fn extract_pattern_name(pattern: &Pattern) -> String {
    match pattern {
        Pattern::Binding { name, .. } => name.clone(),
        Pattern::Wildcard => "_".to_string(),
        Pattern::Literal(lit) => format!("{:?}", lit),
        Pattern::Tuple(elements) => {
            if let Some(first) = elements.first() {
                extract_pattern_name(first)
            } else {
                "_".to_string()
            }
        }
        Pattern::Struct { path, .. } => path.segments.join("::"),
        Pattern::Enum { path, .. } => path.segments.join("::"),
        Pattern::Or(patterns) => {
            if let Some(first) = patterns.first() {
                extract_pattern_name(first)
            } else {
                "_".to_string()
            }
        }
    }
}

/// Report of all dead code found in a module
#[derive(Debug, Clone)]
pub struct DeadCodeReport {
    /// Unused items (functions, types, constants)
    pub unused_items: Vec<UnusedItem>,
    /// Unreachable code sections
    pub unreachable_code: Vec<UnreachableCode>,
    /// Unused imports
    pub unused_imports: Vec<UnusedImport>,
    /// Unused variables/parameters
    pub unused_variables: Vec<UnusedVariable>,
    /// Shadowed definitions
    pub shadowed_definitions: Vec<ShadowedDefinition>,
}

impl DeadCodeReport {
    /// Check if any dead code was found
    pub fn has_issues(&self) -> bool {
        !self.unused_items.is_empty()
            || !self.unreachable_code.is_empty()
            || !self.unused_imports.is_empty()
            || !self.unused_variables.is_empty()
            || !self.shadowed_definitions.is_empty()
    }

    /// Total number of issues
    pub fn issue_count(&self) -> usize {
        self.unused_items.len()
            + self.unreachable_code.len()
            + self.unused_imports.len()
            + self.unused_variables.len()
            + self.shadowed_definitions.len()
    }
}

/// An unused item (function, type, constant)
#[derive(Debug, Clone)]
pub struct UnusedItem {
    /// Name of the unused item
    pub name: String,
    /// Kind of item
    pub kind: ItemKind,
    /// Location in source
    pub span: Span,
    /// Whether it's public (might be used externally)
    pub is_public: bool,
    /// Reason it's considered unused
    pub reason: UnusedReason,
}

/// Kind of item
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ItemKind {
    Function,
    Struct,
    Enum,
    Variant,
    Const,
    Static,
    TypeAlias,
    Trait,
    Impl,
    Module,
}

impl std::fmt::Display for ItemKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ItemKind::Function => write!(f, "function"),
            ItemKind::Struct => write!(f, "struct"),
            ItemKind::Enum => write!(f, "enum"),
            ItemKind::Variant => write!(f, "variant"),
            ItemKind::Const => write!(f, "constant"),
            ItemKind::Static => write!(f, "static"),
            ItemKind::TypeAlias => write!(f, "type alias"),
            ItemKind::Trait => write!(f, "trait"),
            ItemKind::Impl => write!(f, "impl"),
            ItemKind::Module => write!(f, "module"),
        }
    }
}

/// Reason an item is considered unused
#[derive(Debug, Clone)]
pub enum UnusedReason {
    /// Never referenced
    NeverReferenced,
    /// Only referenced by other dead code
    OnlyDeadReferences,
    /// Has #[allow(dead_code)] but still flagged for info
    AllowedButUnused,
}

/// Unreachable code section
#[derive(Debug, Clone)]
pub struct UnreachableCode {
    /// Location of unreachable code
    pub span: Span,
    /// Why it's unreachable
    pub reason: UnreachableReason,
    /// The code that makes it unreachable (return, panic, etc.)
    pub caused_by: Option<Span>,
}

/// Reason code is unreachable
#[derive(Debug, Clone)]
pub enum UnreachableReason {
    /// After return statement
    AfterReturn,
    /// After panic/unreachable
    AfterPanic,
    /// After break/continue
    AfterBreak,
    /// Condition always false
    ConditionAlwaysFalse,
    /// Condition always true (else branch)
    ConditionAlwaysTrue,
    /// Match arm never matches
    NeverMatchingArm,
    /// After infinite loop without break
    AfterInfiniteLoop,
}

impl std::fmt::Display for UnreachableReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnreachableReason::AfterReturn => write!(f, "after return statement"),
            UnreachableReason::AfterPanic => write!(f, "after panic/unreachable"),
            UnreachableReason::AfterBreak => write!(f, "after break/continue"),
            UnreachableReason::ConditionAlwaysFalse => write!(f, "condition is always false"),
            UnreachableReason::ConditionAlwaysTrue => write!(f, "condition is always true"),
            UnreachableReason::NeverMatchingArm => write!(f, "pattern never matches"),
            UnreachableReason::AfterInfiniteLoop => write!(f, "after infinite loop"),
        }
    }
}

/// Unused import
#[derive(Debug, Clone)]
pub struct UnusedImport {
    /// The import path
    pub path: String,
    /// Location of the import
    pub span: Span,
    /// Specific unused items from the import
    pub unused_names: Vec<String>,
}

/// Unused variable or parameter
#[derive(Debug, Clone)]
pub struct UnusedVariable {
    /// Variable name
    pub name: String,
    /// Location of definition
    pub span: Span,
    /// Kind of variable
    pub kind: VariableKind,
    /// Suggestion (prefix with _ to silence)
    pub suggestion: Option<String>,
}

/// Kind of variable
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariableKind {
    Local,
    Parameter,
    LoopVariable,
    MatchBinding,
}

impl std::fmt::Display for VariableKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VariableKind::Local => write!(f, "variable"),
            VariableKind::Parameter => write!(f, "parameter"),
            VariableKind::LoopVariable => write!(f, "loop variable"),
            VariableKind::MatchBinding => write!(f, "match binding"),
        }
    }
}

/// A definition that shadows another
#[derive(Debug, Clone)]
pub struct ShadowedDefinition {
    /// Name being shadowed
    pub name: String,
    /// Location of the shadowing definition
    pub shadow_span: Span,
    /// Location of the original definition
    pub original_span: Span,
    /// Kind of shadowing
    pub kind: ShadowKind,
}

/// Kind of shadowing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadowKind {
    /// Variable shadows another variable
    Variable,
    /// Variable shadows a parameter
    Parameter,
    /// Import shadows a local definition
    Import,
    /// Type shadows another type
    Type,
}

/// Configuration for dead code analysis
#[derive(Debug, Clone)]
pub struct DeadCodeConfig {
    /// Check for unused items
    pub check_unused_items: bool,
    /// Check for unreachable code
    pub check_unreachable: bool,
    /// Check for unused imports
    pub check_unused_imports: bool,
    /// Check for unused variables
    pub check_unused_variables: bool,
    /// Check for shadowing
    pub check_shadowing: bool,
    /// Ignore items prefixed with underscore
    pub ignore_underscore_prefix: bool,
    /// Ignore public items (might be used by external code)
    pub ignore_public_items: bool,
    /// Entry points (functions that are always considered used)
    pub entry_points: Vec<String>,
}

impl Default for DeadCodeConfig {
    fn default() -> Self {
        Self {
            check_unused_items: true,
            check_unreachable: true,
            check_unused_imports: true,
            check_unused_variables: true,
            check_shadowing: true,
            ignore_underscore_prefix: true,
            ignore_public_items: false,
            entry_points: vec!["main".to_string()],
        }
    }
}

/// Dead code analyzer
pub struct DeadCodeAnalyzer {
    config: DeadCodeConfig,
    /// Defined items and their spans
    defined_items: HashMap<String, (ItemKind, Span, bool)>,
    /// Referenced items
    referenced_items: HashSet<String>,
    /// Variable scopes
    variable_scopes: Vec<HashMap<String, (Span, bool, VariableKind)>>,
    /// Collected issues
    unused_items: Vec<UnusedItem>,
    unreachable_code: Vec<UnreachableCode>,
    unused_imports: Vec<UnusedImport>,
    unused_variables: Vec<UnusedVariable>,
    shadowed_definitions: Vec<ShadowedDefinition>,
}

impl DeadCodeAnalyzer {
    /// Create a new analyzer with default config
    pub fn new() -> Self {
        Self::with_config(DeadCodeConfig::default())
    }

    /// Create a new analyzer with custom config
    pub fn with_config(config: DeadCodeConfig) -> Self {
        Self {
            config,
            defined_items: HashMap::new(),
            referenced_items: HashSet::new(),
            variable_scopes: Vec::new(),
            unused_items: Vec::new(),
            unreachable_code: Vec::new(),
            unused_imports: Vec::new(),
            unused_variables: Vec::new(),
            shadowed_definitions: Vec::new(),
        }
    }

    /// Analyze an AST for dead code
    pub fn analyze(&mut self, ast: &Ast) -> DeadCodeReport {
        // First pass: collect all definitions
        self.collect_definitions(ast);

        // Mark entry points as used
        for entry in &self.config.entry_points {
            self.referenced_items.insert(entry.clone());
        }

        // Second pass: collect all references
        self.collect_references(ast);

        // Third pass: find unreachable code
        if self.config.check_unreachable {
            self.find_unreachable_code(ast);
        }

        // Find unused items
        if self.config.check_unused_items {
            self.find_unused_items();
        }

        DeadCodeReport {
            unused_items: std::mem::take(&mut self.unused_items),
            unreachable_code: std::mem::take(&mut self.unreachable_code),
            unused_imports: std::mem::take(&mut self.unused_imports),
            unused_variables: std::mem::take(&mut self.unused_variables),
            shadowed_definitions: std::mem::take(&mut self.shadowed_definitions),
        }
    }

    /// Collect all definitions in the AST
    fn collect_definitions(&mut self, ast: &Ast) {
        for item in &ast.items {
            self.collect_item_definition(item);
        }
    }

    /// Collect definition from an item
    fn collect_item_definition(&mut self, item: &Item) {
        match item {
            Item::Function(f) => {
                let is_public = matches!(f.visibility, Visibility::Public);
                self.defined_items.insert(
                    f.name.clone(),
                    (ItemKind::Function, f.span.clone(), is_public),
                );
            }
            Item::Struct(s) => {
                let is_public = matches!(s.visibility, Visibility::Public);
                self.defined_items.insert(
                    s.name.clone(),
                    (ItemKind::Struct, s.span.clone(), is_public),
                );
            }
            Item::Enum(e) => {
                let is_public = matches!(e.visibility, Visibility::Public);
                self.defined_items
                    .insert(e.name.clone(), (ItemKind::Enum, e.span.clone(), is_public));
                // Also track variants
                for variant in &e.variants {
                    let variant_name = format!("{}::{}", e.name, variant.name);
                    self.defined_items
                        .insert(variant_name, (ItemKind::Variant, e.span.clone(), is_public));
                }
            }
            Item::Global(g) => {
                let is_public = matches!(g.visibility, Visibility::Public);
                let kind = if g.is_mut {
                    ItemKind::Static
                } else {
                    ItemKind::Const
                };
                // Extract name from pattern
                let name = extract_pattern_name(&g.pattern);
                self.defined_items
                    .insert(name, (kind, g.span.clone(), is_public));
            }
            Item::TypeAlias(t) => {
                let is_public = matches!(t.visibility, Visibility::Public);
                self.defined_items.insert(
                    t.name.clone(),
                    (ItemKind::TypeAlias, t.span.clone(), is_public),
                );
            }
            Item::Trait(t) => {
                let is_public = matches!(t.visibility, Visibility::Public);
                self.defined_items
                    .insert(t.name.clone(), (ItemKind::Trait, t.span.clone(), is_public));
            }
            Item::Impl(i) => {
                // Impls are used if their type is used
                if let Some(trait_ref) = &i.trait_ref {
                    self.defined_items.insert(
                        format!(
                            "impl {} for {:?}",
                            trait_ref.segments.join("::"),
                            i.target_type
                        ),
                        (ItemKind::Impl, i.span.clone(), false),
                    );
                }
            }
            Item::Import(_)
            | Item::Export(_)
            | Item::Effect(_)
            | Item::Handler(_)
            | Item::Extern(_)
            | Item::MacroInvocation(_)
            | Item::OntologyImport(_)
            | Item::AlignDecl(_)
            | Item::OdeDef(_)
            | Item::PdeDef(_)
            | Item::CausalModel(_)
            | Item::Module(_) => {}
        }
    }

    /// Collect all references in the AST
    fn collect_references(&mut self, ast: &Ast) {
        for item in &ast.items {
            self.collect_item_references(item);
        }
    }

    /// Collect references from an item
    fn collect_item_references(&mut self, item: &Item) {
        match item {
            Item::Function(f) => {
                self.push_scope();

                // Add parameters to scope via their patterns
                for param in &f.params {
                    self.collect_pattern_bindings(&param.pattern, VariableKind::Parameter);
                    self.collect_type_references(&param.ty);
                }

                // Collect references in body
                self.collect_block_references(&f.body);

                // Check for unused parameters
                if self.config.check_unused_variables {
                    self.check_unused_variables();
                }

                self.pop_scope();
            }
            Item::Struct(s) => {
                // Reference types used in fields
                for field in &s.fields {
                    self.collect_type_references(&field.ty);
                }
            }
            Item::Enum(e) => {
                for variant in &e.variants {
                    match &variant.data {
                        crate::ast::VariantData::Unit => {}
                        crate::ast::VariantData::Tuple(types) => {
                            for ty in types {
                                self.collect_type_references(ty);
                            }
                        }
                        crate::ast::VariantData::Struct(fields) => {
                            for field in fields {
                                self.collect_type_references(&field.ty);
                            }
                        }
                    }
                }
            }
            Item::Global(g) => {
                if let Some(ty) = &g.ty {
                    self.collect_type_references(ty);
                }
                self.collect_expr_references(&g.value);
            }
            Item::TypeAlias(t) => {
                self.collect_type_references(&t.ty);
            }
            Item::Impl(i) => {
                self.collect_type_references(&i.target_type);
                for impl_item in &i.items {
                    match impl_item {
                        crate::ast::ImplItem::Fn(method) => {
                            self.collect_item_references(&Item::Function(method.clone()));
                        }
                        crate::ast::ImplItem::Type(ty_def) => {
                            self.collect_type_references(&ty_def.ty);
                        }
                    }
                }
            }
            Item::Trait(t) => {
                for trait_item in &t.items {
                    match trait_item {
                        crate::ast::TraitItem::Fn(method) => {
                            for param in &method.params {
                                self.collect_type_references(&param.ty);
                            }
                            if let Some(ret) = &method.return_type {
                                self.collect_type_references(ret);
                            }
                        }
                        crate::ast::TraitItem::Type(_) => {}
                    }
                }
            }
            Item::Import(i) => {
                // Track import for unused import detection
                if self.config.check_unused_imports {
                    let _ = i;
                }
            }
            Item::Export(_)
            | Item::Effect(_)
            | Item::Handler(_)
            | Item::Extern(_)
            | Item::MacroInvocation(_)
            | Item::OntologyImport(_)
            | Item::AlignDecl(_)
            | Item::OdeDef(_)
            | Item::PdeDef(_)
            | Item::CausalModel(_)
            | Item::Module(_) => {}
        }
    }

    /// Collect references from a block
    fn collect_block_references(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.collect_stmt_references(stmt);
        }
    }

    /// Collect references from an expression
    fn collect_expr_references(&mut self, expr: &Expr) {
        match expr {
            Expr::Path { path, .. } => {
                let name = path.segments.join("::");
                self.referenced_items.insert(name.clone());
                // Also mark as variable used if it's a simple name
                if path.segments.len() == 1 {
                    self.mark_variable_used(&path.segments[0]);
                }
            }
            Expr::Call { callee, args, .. } => {
                self.collect_expr_references(callee);
                for arg in args {
                    self.collect_expr_references(arg);
                }
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                self.collect_expr_references(receiver);
                self.referenced_items.insert(method.clone());
                for arg in args {
                    self.collect_expr_references(arg);
                }
            }
            Expr::Field { base, .. } => {
                self.collect_expr_references(base);
            }
            Expr::Index { base, index, .. } => {
                self.collect_expr_references(base);
                self.collect_expr_references(index);
            }
            Expr::Binary { left, right, .. } => {
                self.collect_expr_references(left);
                self.collect_expr_references(right);
            }
            Expr::Unary { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.collect_expr_references(condition);
                self.collect_block_references(then_branch);
                if let Some(else_b) = else_branch {
                    self.collect_expr_references(else_b);
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.collect_expr_references(scrutinee);
                for arm in arms {
                    self.push_scope();
                    self.collect_pattern_bindings(&arm.pattern, VariableKind::MatchBinding);
                    if let Some(guard) = &arm.guard {
                        self.collect_expr_references(guard);
                    }
                    self.collect_expr_references(&arm.body);
                    if self.config.check_unused_variables {
                        self.check_unused_variables();
                    }
                    self.pop_scope();
                }
            }
            Expr::Block { block, .. } => {
                self.push_scope();
                self.collect_block_references(block);
                if self.config.check_unused_variables {
                    self.check_unused_variables();
                }
                self.pop_scope();
            }
            Expr::Loop { body, .. } => {
                self.collect_block_references(body);
            }
            Expr::While {
                condition, body, ..
            } => {
                self.collect_expr_references(condition);
                self.collect_block_references(body);
            }
            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => {
                self.collect_expr_references(iter);
                self.push_scope();
                self.collect_pattern_bindings(pattern, VariableKind::LoopVariable);
                self.collect_block_references(body);
                if self.config.check_unused_variables {
                    self.check_unused_variables();
                }
                self.pop_scope();
            }
            Expr::Return { value, .. } => {
                if let Some(v) = value {
                    self.collect_expr_references(v);
                }
            }
            Expr::Break { value, .. } => {
                if let Some(v) = value {
                    self.collect_expr_references(v);
                }
            }
            Expr::Array { elements, .. } => {
                for elem in elements {
                    self.collect_expr_references(elem);
                }
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.collect_expr_references(s);
                }
                if let Some(e) = end {
                    self.collect_expr_references(e);
                }
            }
            Expr::Tuple { elements, .. } => {
                for elem in elements {
                    self.collect_expr_references(elem);
                }
            }
            Expr::StructLit { path, fields, .. } => {
                self.referenced_items.insert(path.segments.join("::"));
                for (_, value) in fields {
                    self.collect_expr_references(value);
                }
            }
            Expr::Closure { params, body, .. } => {
                self.push_scope();
                for (name, _ty) in params {
                    self.variable_scopes.last_mut().unwrap().insert(
                        name.clone(),
                        (Span::default(), false, VariableKind::Parameter),
                    );
                }
                self.collect_expr_references(body);
                if self.config.check_unused_variables {
                    self.check_unused_variables();
                }
                self.pop_scope();
            }
            Expr::Cast { expr, ty, .. } => {
                self.collect_expr_references(expr);
                self.collect_type_references(ty);
            }
            Expr::Try { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Expr::Await { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Expr::Perform { effect, args, .. } => {
                self.referenced_items.insert(effect.segments.join("::"));
                for arg in args {
                    self.collect_expr_references(arg);
                }
            }
            Expr::Handle { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Expr::Sample { distribution, .. } => {
                self.collect_expr_references(distribution);
            }
            Expr::AsyncBlock { block, .. } => {
                self.collect_block_references(block);
            }
            Expr::AsyncClosure { params, body, .. } => {
                self.push_scope();
                for (name, _ty) in params {
                    self.variable_scopes.last_mut().unwrap().insert(
                        name.clone(),
                        (Span::default(), false, VariableKind::Parameter),
                    );
                }
                self.collect_expr_references(body);
                if self.config.check_unused_variables {
                    self.check_unused_variables();
                }
                self.pop_scope();
            }
            Expr::Spawn { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Expr::Select { arms, .. } => {
                for arm in arms {
                    self.collect_expr_references(&arm.future);
                    self.push_scope();
                    self.collect_pattern_bindings(&arm.pattern, VariableKind::MatchBinding);
                    if let Some(guard) = &arm.guard {
                        self.collect_expr_references(guard);
                    }
                    self.collect_expr_references(&arm.body);
                    if self.config.check_unused_variables {
                        self.check_unused_variables();
                    }
                    self.pop_scope();
                }
            }
            Expr::Join { futures, .. } => {
                for future in futures {
                    self.collect_expr_references(future);
                }
            }
            Expr::TupleField { base, .. } => {
                self.collect_expr_references(base);
            }
            // Literals and other simple expressions have no references
            Expr::Literal { .. } | Expr::Continue { .. } | Expr::MacroInvocation(_) => {}

            // Epistemic expressions
            Expr::Do { interventions, .. } => {
                for (_, value) in interventions {
                    self.collect_expr_references(value);
                }
            }
            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                self.collect_expr_references(factual);
                self.collect_expr_references(intervention);
                self.collect_expr_references(outcome);
            }
            Expr::KnowledgeExpr {
                value,
                epsilon,
                validity,
                provenance,
                ..
            } => {
                self.collect_expr_references(value);
                if let Some(e) = epsilon {
                    self.collect_expr_references(e);
                }
                if let Some(v) = validity {
                    self.collect_expr_references(v);
                }
                if let Some(p) = provenance {
                    self.collect_expr_references(p);
                }
            }
            Expr::Uncertain {
                value, uncertainty, ..
            } => {
                self.collect_expr_references(value);
                self.collect_expr_references(uncertainty);
            }
            Expr::GpuAnnotated { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Expr::Observe {
                data, distribution, ..
            } => {
                self.collect_expr_references(data);
                self.collect_expr_references(distribution);
            }
            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                self.collect_expr_references(target);
                for g in given {
                    self.collect_expr_references(g);
                }
                for (_, value) in interventions {
                    self.collect_expr_references(value);
                }
            }
            Expr::OntologyTerm { ontology, term, .. } => {
                // Track ontology term as a reference
                self.referenced_items
                    .insert(format!("{}:{}", ontology, term));
            }
        }
    }

    /// Collect references from a statement
    fn collect_stmt_references(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                pattern, ty, value, ..
            } => {
                if let Some(t) = ty {
                    self.collect_type_references(t);
                }
                if let Some(v) = value {
                    self.collect_expr_references(v);
                }
                self.collect_pattern_bindings(pattern, VariableKind::Local);
            }
            Stmt::Expr { expr, .. } => {
                self.collect_expr_references(expr);
            }
            Stmt::Assign { target, value, .. } => {
                self.collect_expr_references(target);
                self.collect_expr_references(value);
            }
            Stmt::Empty | Stmt::MacroInvocation(_) => {}
        }
    }

    /// Collect references from a type expression
    fn collect_type_references(&mut self, ty: &TypeExpr) {
        match ty {
            TypeExpr::Named { path, args, .. } => {
                // Reference the type name
                if let Some(name) = path.segments.last() {
                    self.referenced_items.insert(name.clone());
                }
                // Also reference full path
                self.referenced_items.insert(path.segments.join("::"));
                for arg in args {
                    self.collect_type_references(arg);
                }
            }
            TypeExpr::Array { element, .. } => {
                self.collect_type_references(element);
            }
            TypeExpr::Tuple(elements) => {
                for elem in elements {
                    self.collect_type_references(elem);
                }
            }
            TypeExpr::Function {
                params,
                return_type,
                ..
            } => {
                for param in params {
                    self.collect_type_references(param);
                }
                self.collect_type_references(return_type);
            }
            TypeExpr::Reference { inner, .. } => {
                self.collect_type_references(inner);
            }
            TypeExpr::RawPointer { inner, .. } => {
                self.collect_type_references(inner);
            }
            TypeExpr::Unit | TypeExpr::SelfType | TypeExpr::Infer => {}

            // Epistemic types
            TypeExpr::Knowledge { value_type, .. } => {
                self.collect_type_references(value_type);
            }
            TypeExpr::Quantity { numeric_type, .. } => {
                self.collect_type_references(numeric_type);
            }
            TypeExpr::Tensor { element_type, .. } => {
                self.collect_type_references(element_type);
            }
            TypeExpr::Ontology { .. } => {}
            TypeExpr::Linear { inner, .. } => {
                self.collect_type_references(inner);
            }
            TypeExpr::Effected { inner, .. } => {
                self.collect_type_references(inner);
            }
            TypeExpr::Tile { element_type, .. } => {
                self.collect_type_references(element_type);
            }
            TypeExpr::Refinement { base_type, .. } => {
                self.collect_type_references(base_type);
            }
        }
    }

    /// Collect variable bindings from a pattern
    fn collect_pattern_bindings(&mut self, pattern: &Pattern, kind: VariableKind) {
        match pattern {
            Pattern::Binding { name, .. } => {
                self.add_variable(name, Span::default(), kind);
            }
            Pattern::Tuple(patterns) => {
                for p in patterns {
                    self.collect_pattern_bindings(p, kind);
                }
            }
            Pattern::Struct { path, fields, .. } => {
                if let Some(name) = path.segments.last() {
                    self.referenced_items.insert(name.clone());
                }
                for (_, pat) in fields {
                    self.collect_pattern_bindings(pat, kind);
                }
            }
            Pattern::Enum { path, patterns, .. } => {
                if let Some(name) = path.segments.last() {
                    self.referenced_items.insert(name.clone());
                }
                if let Some(pats) = patterns {
                    for p in pats {
                        self.collect_pattern_bindings(p, kind);
                    }
                }
            }
            Pattern::Or(patterns) => {
                for p in patterns {
                    self.collect_pattern_bindings(p, kind);
                }
            }
            Pattern::Wildcard | Pattern::Literal(_) => {}
        }
    }

    /// Find unreachable code in the AST
    fn find_unreachable_code(&mut self, ast: &Ast) {
        for item in &ast.items {
            if let Item::Function(f) = item {
                self.find_unreachable_in_block(&f.body);
            }
        }
    }

    /// Find unreachable code in a block
    fn find_unreachable_in_block(&mut self, block: &Block) {
        let mut terminated = false;
        let mut terminator_span = None;
        let mut terminator_reason = None;

        for stmt in &block.stmts {
            if terminated {
                // This statement is unreachable
                let stmt_span = self.stmt_span(stmt);
                self.unreachable_code.push(UnreachableCode {
                    span: stmt_span,
                    reason: terminator_reason.clone().unwrap(),
                    caused_by: terminator_span.clone(),
                });
                break; // Only report first unreachable
            }

            // Check if this statement terminates
            if let Some((reason, span)) = self.is_terminating_stmt(stmt) {
                terminated = true;
                terminator_span = Some(span);
                terminator_reason = Some(reason);
            }

            // Recurse into statement
            self.find_unreachable_in_stmt(stmt);
        }
    }

    /// Find unreachable code in a statement
    fn find_unreachable_in_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Expr { expr, .. } => {
                self.find_unreachable_in_expr(expr);
            }
            Stmt::Let { value, .. } => {
                if let Some(v) = value {
                    self.find_unreachable_in_expr(v);
                }
            }
            Stmt::Assign { target, value, .. } => {
                self.find_unreachable_in_expr(target);
                self.find_unreachable_in_expr(value);
            }
            Stmt::Empty | Stmt::MacroInvocation(_) => {}
        }
    }

    /// Find unreachable code in an expression
    fn find_unreachable_in_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Block { block, .. } => {
                self.find_unreachable_in_block(block);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                // Check for constant conditions
                if let Some(value) = self.is_constant_bool(condition) {
                    if value {
                        // Else branch is unreachable
                        if let Some(_else_b) = else_branch {
                            // Note: would need span for else block
                        }
                    }
                }

                self.find_unreachable_in_expr(condition);
                self.find_unreachable_in_block(then_branch);
                if let Some(else_b) = else_branch {
                    self.find_unreachable_in_expr(else_b);
                }
            }
            Expr::Loop { body, .. } => {
                self.find_unreachable_in_block(body);
            }
            Expr::While {
                condition, body, ..
            } => {
                self.find_unreachable_in_expr(condition);
                self.find_unreachable_in_block(body);
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.find_unreachable_in_expr(scrutinee);
                for arm in arms {
                    self.find_unreachable_in_expr(&arm.body);
                }
            }
            Expr::Call { callee, args, .. } => {
                self.find_unreachable_in_expr(callee);
                for arg in args {
                    self.find_unreachable_in_expr(arg);
                }
            }
            Expr::Binary { left, right, .. } => {
                self.find_unreachable_in_expr(left);
                self.find_unreachable_in_expr(right);
            }
            Expr::Unary { expr, .. } => {
                self.find_unreachable_in_expr(expr);
            }
            _ => {}
        }
    }

    /// Check if a statement terminates control flow
    fn is_terminating_stmt(&self, stmt: &Stmt) -> Option<(UnreachableReason, Span)> {
        match stmt {
            Stmt::Expr { expr, .. } => self.is_terminating_expr(expr),
            _ => None,
        }
    }

    /// Check if an expression terminates control flow
    fn is_terminating_expr(&self, expr: &Expr) -> Option<(UnreachableReason, Span)> {
        match expr {
            Expr::Return { .. } => Some((UnreachableReason::AfterReturn, Span::dummy())),
            Expr::Break { .. } => Some((UnreachableReason::AfterBreak, Span::dummy())),
            Expr::Continue { .. } => Some((UnreachableReason::AfterBreak, Span::dummy())),
            Expr::Call { callee, .. } => {
                // Check for panic calls
                if let Expr::Path { path, .. } = callee.as_ref() {
                    if let Some(name) = path.segments.last() {
                        if name == "panic" || name == "unreachable" || name == "todo" {
                            return Some((UnreachableReason::AfterPanic, Span::dummy()));
                        }
                    }
                }
                None
            }
            Expr::Loop { body, .. } => {
                // Infinite loop without break terminates
                if !self.has_break_in_block(body) {
                    Some((UnreachableReason::AfterInfiniteLoop, Span::dummy()))
                } else {
                    None
                }
            }
            Expr::Block { block, .. } => {
                // Check last statement
                if let Some(last) = block.stmts.last() {
                    return self.is_terminating_stmt(last);
                }
                None
            }
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                // If both branches terminate, the if terminates
                let then_term = block_stmts_terminate(then_branch, self);
                let else_term = else_branch
                    .as_ref()
                    .and_then(|e| self.is_terminating_expr(e));

                if then_term.is_some() && else_term.is_some() {
                    then_term
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Check if block contains a break statement
    fn has_break_in_block(&self, block: &Block) -> bool {
        for stmt in &block.stmts {
            if self.has_break_in_stmt(stmt) {
                return true;
            }
        }
        false
    }

    fn has_break_in_stmt(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr { expr, .. } => self.has_break_in_expr(expr),
            _ => false,
        }
    }

    fn has_break_in_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Break { .. } => true,
            Expr::Block { block, .. } => self.has_break_in_block(block),
            Expr::If {
                then_branch,
                else_branch,
                ..
            } => {
                self.has_break_in_block(then_branch)
                    || else_branch
                        .as_ref()
                        .map(|e| self.has_break_in_expr(e))
                        .unwrap_or(false)
            }
            Expr::Match { arms, .. } => arms.iter().any(|a| self.has_break_in_expr(&a.body)),
            // Don't recurse into nested loops
            Expr::Loop { .. } | Expr::While { .. } | Expr::For { .. } => false,
            _ => false,
        }
    }

    /// Check if expression is a constant boolean
    fn is_constant_bool(&self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::Literal { value, .. } => {
                if let crate::ast::Literal::Bool(b) = value {
                    Some(*b)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Find unused items
    fn find_unused_items(&mut self) {
        for (name, (kind, span, is_public)) in &self.defined_items {
            // Skip underscore-prefixed names
            if self.config.ignore_underscore_prefix && name.starts_with('_') {
                continue;
            }

            // Skip public items if configured
            if self.config.ignore_public_items && *is_public {
                continue;
            }

            // Check if referenced
            if !self.referenced_items.contains(name) {
                self.unused_items.push(UnusedItem {
                    name: name.clone(),
                    kind: *kind,
                    span: span.clone(),
                    is_public: *is_public,
                    reason: UnusedReason::NeverReferenced,
                });
            }
        }
    }

    // Variable scope management

    fn push_scope(&mut self) {
        self.variable_scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.variable_scopes.pop();
    }

    fn add_variable(&mut self, name: &str, span: Span, kind: VariableKind) {
        // Check for shadowing
        if self.config.check_shadowing {
            for scope in self.variable_scopes.iter().rev() {
                if let Some((original_span, _, _)) = scope.get(name) {
                    self.shadowed_definitions.push(ShadowedDefinition {
                        name: name.to_string(),
                        shadow_span: span.clone(),
                        original_span: original_span.clone(),
                        kind: ShadowKind::Variable,
                    });
                    break;
                }
            }
        }

        if let Some(scope) = self.variable_scopes.last_mut() {
            scope.insert(name.to_string(), (span, false, kind));
        }
    }

    fn mark_variable_used(&mut self, name: &str) {
        for scope in self.variable_scopes.iter_mut().rev() {
            if let Some((_, used, _)) = scope.get_mut(name) {
                *used = true;
                return;
            }
        }
    }

    fn check_unused_variables(&mut self) {
        if let Some(scope) = self.variable_scopes.last() {
            for (name, (span, used, kind)) in scope {
                if !used {
                    // Skip underscore-prefixed names
                    if self.config.ignore_underscore_prefix && name.starts_with('_') {
                        continue;
                    }

                    let suggestion = if !name.starts_with('_') {
                        Some(format!("_{}", name))
                    } else {
                        None
                    };

                    self.unused_variables.push(UnusedVariable {
                        name: name.clone(),
                        span: span.clone(),
                        kind: *kind,
                        suggestion,
                    });
                }
            }
        }
    }

    // Span helpers

    fn stmt_span(&self, _stmt: &Stmt) -> Span {
        // Stmt doesn't have span info, return dummy
        Span::dummy()
    }
}

fn block_stmts_terminate(
    block: &Block,
    analyzer: &DeadCodeAnalyzer,
) -> Option<(UnreachableReason, Span)> {
    if let Some(last) = block.stmts.last() {
        analyzer.is_terminating_stmt(last)
    } else {
        None
    }
}

impl Default for DeadCodeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Analyze an AST for dead code with default configuration
pub fn analyze_dead_code(ast: &Ast) -> DeadCodeReport {
    let mut analyzer = DeadCodeAnalyzer::new();
    analyzer.analyze(ast)
}

/// Analyze an AST for dead code with custom configuration
pub fn analyze_dead_code_with_config(ast: &Ast, config: DeadCodeConfig) -> DeadCodeReport {
    let mut analyzer = DeadCodeAnalyzer::with_config(config);
    analyzer.analyze(ast)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span() -> Span {
        Span { start: 0, end: 0 }
    }

    #[test]
    fn test_dead_code_report_empty() {
        let report = DeadCodeReport {
            unused_items: vec![],
            unreachable_code: vec![],
            unused_imports: vec![],
            unused_variables: vec![],
            shadowed_definitions: vec![],
        };

        assert!(!report.has_issues());
        assert_eq!(report.issue_count(), 0);
    }

    #[test]
    fn test_dead_code_report_with_issues() {
        let report = DeadCodeReport {
            unused_items: vec![UnusedItem {
                name: "foo".to_string(),
                kind: ItemKind::Function,
                span: make_span(),
                is_public: false,
                reason: UnusedReason::NeverReferenced,
            }],
            unreachable_code: vec![],
            unused_imports: vec![],
            unused_variables: vec![UnusedVariable {
                name: "x".to_string(),
                span: make_span(),
                kind: VariableKind::Local,
                suggestion: Some("_x".to_string()),
            }],
            shadowed_definitions: vec![],
        };

        assert!(report.has_issues());
        assert_eq!(report.issue_count(), 2);
    }

    #[test]
    fn test_item_kind_display() {
        assert_eq!(format!("{}", ItemKind::Function), "function");
        assert_eq!(format!("{}", ItemKind::Struct), "struct");
        assert_eq!(format!("{}", ItemKind::Enum), "enum");
    }

    #[test]
    fn test_unreachable_reason_display() {
        assert_eq!(
            format!("{}", UnreachableReason::AfterReturn),
            "after return statement"
        );
        assert_eq!(
            format!("{}", UnreachableReason::ConditionAlwaysFalse),
            "condition is always false"
        );
    }

    #[test]
    fn test_default_config() {
        let config = DeadCodeConfig::default();
        assert!(config.check_unused_items);
        assert!(config.check_unreachable);
        assert!(config.ignore_underscore_prefix);
        assert!(config.entry_points.contains(&"main".to_string()));
    }
}
