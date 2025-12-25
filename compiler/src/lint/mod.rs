//! Linter framework for Sounio
//!
//! Provides static analysis and code quality checks with:
//! - Built-in lint rules for correctness, style, and performance
//! - Configurable lint levels (allow, warn, deny, forbid)
//! - Auto-fix suggestions
//! - Custom lint plugin support

pub mod config;
pub mod rules;

pub use config::LintConfig;

use std::collections::{HashMap, HashSet};

use crate::ast::{
    self, Ast, Block, Expr, ImplItem, Item, Pattern, Stmt, TraitItem, TypeExpr, VariantData,
};
use crate::common::Span;

/// Lint level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum LintLevel {
    /// Lint is allowed (disabled)
    Allow,

    /// Lint produces a warning
    #[default]
    Warn,

    /// Lint produces an error
    Deny,

    /// Lint produces an error and cannot be overridden
    Forbid,
}

impl std::fmt::Display for LintLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LintLevel::Allow => write!(f, "allow"),
            LintLevel::Warn => write!(f, "warn"),
            LintLevel::Deny => write!(f, "deny"),
            LintLevel::Forbid => write!(f, "forbid"),
        }
    }
}

/// Lint category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LintCategory {
    /// Correctness issues (likely bugs)
    Correctness,

    /// Style issues
    Style,

    /// Performance issues
    Performance,

    /// Complexity issues
    Complexity,

    /// Security issues
    Security,

    /// Documentation issues
    Documentation,

    /// Deprecated features
    Deprecation,

    /// Effect-related issues
    Effects,

    /// Memory safety issues
    Safety,

    /// Naming conventions
    Naming,
}

impl std::fmt::Display for LintCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LintCategory::Correctness => write!(f, "correctness"),
            LintCategory::Style => write!(f, "style"),
            LintCategory::Performance => write!(f, "performance"),
            LintCategory::Complexity => write!(f, "complexity"),
            LintCategory::Security => write!(f, "security"),
            LintCategory::Documentation => write!(f, "documentation"),
            LintCategory::Deprecation => write!(f, "deprecation"),
            LintCategory::Effects => write!(f, "effects"),
            LintCategory::Safety => write!(f, "safety"),
            LintCategory::Naming => write!(f, "naming"),
        }
    }
}

/// A lint rule
pub trait Lint: Send + Sync {
    /// Unique identifier for this lint
    fn id(&self) -> &'static str;

    /// Human-readable name
    fn name(&self) -> &'static str;

    /// Description
    fn description(&self) -> &'static str;

    /// Default level
    fn default_level(&self) -> LintLevel;

    /// Category
    fn category(&self) -> LintCategory;

    /// Check an item
    fn check_item(&self, _item: &Item, _ctx: &mut LintContext) {}

    /// Check a statement
    fn check_stmt(&self, _stmt: &Stmt, _ctx: &mut LintContext) {}

    /// Check an expression
    fn check_expr(&self, _expr: &Expr, _ctx: &mut LintContext) {}

    /// Check a pattern
    fn check_pattern(&self, _pattern: &Pattern, _ctx: &mut LintContext) {}

    /// Check a type
    fn check_type(&self, _ty: &TypeExpr, _ctx: &mut LintContext) {}

    /// Check the whole AST (for cross-cutting concerns)
    fn check_crate(&self, _ast: &Ast, _ctx: &mut LintContext) {}

    /// Provide a fix if possible
    fn fix(&self, _diagnostic: &LintDiagnostic) -> Option<Fix> {
        None
    }
}

/// Lint diagnostic
#[derive(Debug, Clone)]
pub struct LintDiagnostic {
    /// Lint ID
    pub lint_id: String,

    /// Lint name (alias for lint_id for convenience)
    pub lint_name: String,

    /// Level
    pub level: LintLevel,

    /// Message
    pub message: String,

    /// Primary span
    pub span: Span,

    /// Secondary labels
    pub labels: Vec<(Span, String)>,

    /// Help messages
    pub help: Vec<String>,

    /// Notes
    pub notes: Vec<String>,

    /// Suggested fixes
    pub suggestions: Vec<Suggestion>,
}

impl LintDiagnostic {
    /// Create a new diagnostic
    pub fn new(
        lint_id: impl Into<String>,
        level: LintLevel,
        message: impl Into<String>,
        span: Span,
    ) -> Self {
        let id = lint_id.into();
        Self {
            lint_name: id.clone(),
            lint_id: id,
            level,
            message: message.into(),
            span,
            labels: Vec::new(),
            help: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }
}

/// A suggested fix
#[derive(Debug, Clone)]
pub struct Suggestion {
    /// Description
    pub message: String,

    /// Edits to apply
    pub edits: Vec<Edit>,

    /// Applicability
    pub applicability: Applicability,
}

/// Edit applicability
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Applicability {
    /// Can be applied automatically
    MachineApplicable,

    /// Has placeholders that need to be filled
    HasPlaceholders,

    /// May change semantics
    MaybeIncorrect,

    /// Unknown applicability
    Unspecified,
}

/// A text edit
#[derive(Debug, Clone)]
pub struct Edit {
    pub span: Span,
    pub replacement: String,
}

/// A code fix
#[derive(Debug, Clone)]
pub struct Fix {
    /// Description
    pub message: String,

    /// Edits to apply
    pub edits: Vec<Edit>,
}

/// Context for lint checking
pub struct LintContext<'a> {
    /// Current file path
    pub file: &'a str,

    /// Source code
    pub source: &'a str,

    /// Diagnostics collected
    pub diagnostics: Vec<LintDiagnostic>,

    /// Lint levels (overrides)
    pub levels: HashMap<String, LintLevel>,

    /// Current function (for context)
    pub current_function: Option<String>,

    /// Scope stack for variable tracking
    scope_stack: Vec<Scope>,

    /// Used names (for dead code detection)
    used_names: HashSet<String>,

    /// Defined names with spans
    defined_names: HashMap<String, (Span, NameKind)>,
}

#[derive(Debug)]
struct Scope {
    variables: HashMap<String, VarInfo>,
}

#[derive(Debug, Clone)]
struct VarInfo {
    span: Span,
    ty: Option<TypeExpr>,
    used: bool,
    mutable: bool,
    assigned: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum NameKind {
    Function,
    Type,
    Variable,
    Const,
    Module,
}

impl<'a> LintContext<'a> {
    /// Create new context
    pub fn new(file: &'a str, source: &'a str) -> Self {
        LintContext {
            file,
            source,
            diagnostics: Vec::new(),
            levels: HashMap::new(),
            current_function: None,
            scope_stack: vec![Scope {
                variables: HashMap::new(),
            }],
            used_names: HashSet::new(),
            defined_names: HashMap::new(),
        }
    }

    /// Report a lint diagnostic
    pub fn report(&mut self, lint: &dyn Lint, message: impl Into<String>, span: Span) {
        let level = self
            .levels
            .get(lint.id())
            .copied()
            .unwrap_or(lint.default_level());

        if level == LintLevel::Allow {
            return;
        }

        let diagnostic = LintDiagnostic {
            lint_id: lint.id().to_string(),
            lint_name: lint.name().to_string(),
            level,
            message: message.into(),
            span,
            labels: Vec::new(),
            help: Vec::new(),
            notes: vec![format!(
                "lint `{}` ({}) in category `{}`",
                lint.id(),
                lint.name(),
                lint.category()
            )],
            suggestions: Vec::new(),
        };

        self.diagnostics.push(diagnostic);
    }

    /// Report with help message
    pub fn report_with_help(
        &mut self,
        lint: &dyn Lint,
        message: impl Into<String>,
        span: Span,
        help: impl Into<String>,
    ) {
        let level = self
            .levels
            .get(lint.id())
            .copied()
            .unwrap_or(lint.default_level());

        if level == LintLevel::Allow {
            return;
        }

        let diagnostic = LintDiagnostic {
            lint_id: lint.id().to_string(),
            lint_name: lint.name().to_string(),
            level,
            message: message.into(),
            span,
            labels: Vec::new(),
            help: vec![help.into()],
            notes: vec![format!(
                "lint `{}` ({}) in category `{}`",
                lint.id(),
                lint.name(),
                lint.category()
            )],
            suggestions: Vec::new(),
        };

        self.diagnostics.push(diagnostic);
    }

    /// Report with suggestion
    pub fn report_with_suggestion(
        &mut self,
        lint: &dyn Lint,
        message: impl Into<String>,
        span: Span,
        suggestion: Suggestion,
    ) {
        let level = self
            .levels
            .get(lint.id())
            .copied()
            .unwrap_or(lint.default_level());

        if level == LintLevel::Allow {
            return;
        }

        let diagnostic = LintDiagnostic {
            lint_id: lint.id().to_string(),
            lint_name: lint.name().to_string(),
            level,
            message: message.into(),
            span,
            labels: Vec::new(),
            help: Vec::new(),
            notes: vec![format!(
                "lint `{}` ({}) in category `{}`",
                lint.id(),
                lint.name(),
                lint.category()
            )],
            suggestions: vec![suggestion],
        };

        self.diagnostics.push(diagnostic);
    }

    /// Enter a new scope
    pub fn enter_scope(&mut self) {
        self.scope_stack.push(Scope {
            variables: HashMap::new(),
        });
    }

    /// Exit current scope
    pub fn exit_scope(&mut self) {
        self.scope_stack.pop();
    }

    /// Declare a variable
    pub fn declare_var(&mut self, name: &str, span: Span, ty: Option<TypeExpr>, mutable: bool) {
        if let Some(scope) = self.scope_stack.last_mut() {
            scope.variables.insert(
                name.to_string(),
                VarInfo {
                    span,
                    ty,
                    used: false,
                    mutable,
                    assigned: false,
                },
            );
        }
    }

    /// Mark variable as used
    pub fn use_var(&mut self, name: &str) {
        for scope in self.scope_stack.iter_mut().rev() {
            if let Some(var) = scope.variables.get_mut(name) {
                var.used = true;
                return;
            }
        }
        // Also track for cross-file analysis
        self.used_names.insert(name.to_string());
    }

    /// Mark variable as assigned
    pub fn assign_var(&mut self, name: &str) {
        for scope in self.scope_stack.iter_mut().rev() {
            if let Some(var) = scope.variables.get_mut(name) {
                var.assigned = true;
                return;
            }
        }
    }

    /// Get unused variables in current scope
    pub fn unused_vars(&self) -> Vec<(&str, Span)> {
        self.scope_stack
            .last()
            .map(|scope| {
                scope
                    .variables
                    .iter()
                    .filter(|(name, info)| !info.used && !name.starts_with('_'))
                    .map(|(name, info)| (name.as_str(), info.span))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if variable is mutable
    pub fn is_var_mutable(&self, name: &str) -> Option<bool> {
        for scope in self.scope_stack.iter().rev() {
            if let Some(var) = scope.variables.get(name) {
                return Some(var.mutable);
            }
        }
        None
    }

    /// Define a name (for dead code detection)
    pub fn define_name(&mut self, name: &str, span: Span, kind: NameKind) {
        self.defined_names.insert(name.to_string(), (span, kind));
    }

    /// Mark a name as used
    pub fn use_name(&mut self, name: &str) {
        self.used_names.insert(name.to_string());
    }

    /// Get unused names
    pub fn unused_names(&self) -> Vec<(&str, Span, NameKind)> {
        self.defined_names
            .iter()
            .filter(|(name, _)| !self.used_names.contains(*name) && !name.starts_with('_'))
            .map(|(name, (span, kind))| (name.as_str(), *span, *kind))
            .collect()
    }
}

/// Linter engine
pub struct Linter {
    /// Registered lints
    lints: Vec<Box<dyn Lint>>,

    /// Lint configuration
    config: config::LintConfig,
}

impl Linter {
    /// Create a new linter with default rules
    pub fn new() -> Self {
        let mut linter = Linter {
            lints: Vec::new(),
            config: config::LintConfig::default(),
        };

        linter.register_builtin_lints();
        linter
    }

    /// Create with custom config
    pub fn with_config(config: config::LintConfig) -> Self {
        let mut linter = Linter {
            lints: Vec::new(),
            config,
        };

        linter.register_builtin_lints();
        linter
    }

    /// Register a lint rule
    pub fn register(&mut self, lint: Box<dyn Lint>) {
        self.lints.push(lint);
    }

    /// Register built-in lints
    fn register_builtin_lints(&mut self) {
        use rules::*;

        // Correctness
        self.register(Box::new(UnusedVariable));
        self.register(Box::new(UnusedImport));
        self.register(Box::new(UnusedFunction));
        self.register(Box::new(UnreachableCode));
        self.register(Box::new(DivisionByZero));
        self.register(Box::new(InfiniteLoop));

        // Style
        self.register(Box::new(NamingConvention));
        self.register(Box::new(MissingDocumentation));

        // Complexity
        self.register(Box::new(TooManyArguments));
        self.register(Box::new(TooLongFunction));
        self.register(Box::new(DeepNesting));
        self.register(Box::new(CyclomaticComplexity));

        // Performance
        self.register(Box::new(UnnecessaryClone));
        self.register(Box::new(LargeStackValue));

        // Effects
        self.register(Box::new(UnusedEffect));
        self.register(Box::new(MissingEffectAnnotation));

        // Safety
        self.register(Box::new(UnsafeBlock));
    }

    /// Get all registered lints
    pub fn lints(&self) -> &[Box<dyn Lint>] {
        &self.lints
    }

    /// Get lint by ID
    pub fn get_lint(&self, id: &str) -> Option<&dyn Lint> {
        self.lints.iter().find(|l| l.id() == id).map(|l| l.as_ref())
    }

    /// Run lints on AST
    pub fn lint(&self, ast: &ast::Ast, file: &str, source: &str) -> Vec<LintDiagnostic> {
        let mut ctx = LintContext::new(file, source);

        // Apply config overrides
        for (id, level) in &self.config.levels {
            ctx.levels.insert(id.clone(), *level);
        }

        // Check crate-level lints
        for lint in &self.lints {
            if self.is_lint_enabled(lint.as_ref(), &ctx) {
                lint.check_crate(ast, &mut ctx);
            }
        }

        // Visit AST
        for item in &ast.items {
            self.lint_item(item, &mut ctx);
        }

        ctx.diagnostics
    }

    /// Run lints on an Ast (simplified interface for CLI)
    pub fn lint_ast(&self, ast: &Ast) -> Vec<LintDiagnostic> {
        // Create a simplified context
        let mut ctx = LintContext::new("", "");

        // Apply config overrides
        for (id, level) in &self.config.levels {
            ctx.levels.insert(id.clone(), *level);
        }

        // Visit module items
        for item in &ast.items {
            self.lint_module_item(item, &mut ctx);
        }

        ctx.diagnostics
    }

    /// Lint a module item (for simplified Module-based linting)
    fn lint_module_item(&self, item: &ast::Item, ctx: &mut LintContext) {
        match item {
            ast::Item::Function(f) => {
                // Check naming convention
                if !f.name.starts_with('_') && !self.is_snake_case(&f.name) {
                    let mut diag = LintDiagnostic::new(
                        "naming_convention",
                        self.config
                            .get_level("naming_convention")
                            .unwrap_or(LintLevel::Warn),
                        format!("function `{}` should be snake_case", f.name),
                        f.span,
                    );
                    diag.help
                        .push(format!("rename to `{}`", self.to_snake_case(&f.name)));
                    ctx.diagnostics.push(diag);
                }

                // Check parameter count
                if f.params.len() > self.config.max_params {
                    let mut diag = LintDiagnostic::new(
                        "too_many_arguments",
                        self.config
                            .get_level("too_many_arguments")
                            .unwrap_or(LintLevel::Warn),
                        format!(
                            "function `{}` has {} parameters (max: {})",
                            f.name,
                            f.params.len(),
                            self.config.max_params
                        ),
                        f.span,
                    );
                    diag.help
                        .push("consider grouping parameters into a struct".to_string());
                    ctx.diagnostics.push(diag);
                }
            }
            ast::Item::Struct(s) => {
                // Check naming convention
                if !self.is_pascal_case(&s.name) {
                    let mut diag = LintDiagnostic::new(
                        "naming_convention",
                        self.config
                            .get_level("naming_convention")
                            .unwrap_or(LintLevel::Warn),
                        format!("struct `{}` should be PascalCase", s.name),
                        s.span,
                    );
                    diag.help
                        .push(format!("rename to `{}`", self.to_pascal_case(&s.name)));
                    ctx.diagnostics.push(diag);
                }
            }
            ast::Item::Enum(e) => {
                // Check naming convention
                if !self.is_pascal_case(&e.name) {
                    let mut diag = LintDiagnostic::new(
                        "naming_convention",
                        self.config
                            .get_level("naming_convention")
                            .unwrap_or(LintLevel::Warn),
                        format!("enum `{}` should be PascalCase", e.name),
                        e.span,
                    );
                    diag.help
                        .push(format!("rename to `{}`", self.to_pascal_case(&e.name)));
                    ctx.diagnostics.push(diag);
                }
            }
            ast::Item::Global(g) if g.is_const => {
                // Check naming convention (SCREAMING_SNAKE_CASE) for constants
                if let Pattern::Binding { name, .. } = &g.pattern
                    && !self.is_screaming_snake_case(name)
                {
                    let mut diag = LintDiagnostic::new(
                        "naming_convention",
                        self.config
                            .get_level("naming_convention")
                            .unwrap_or(LintLevel::Warn),
                        format!("constant `{}` should be SCREAMING_SNAKE_CASE", name),
                        g.span,
                    );
                    diag.help.push(format!(
                        "rename to `{}`",
                        self.to_screaming_snake_case(name)
                    ));
                    ctx.diagnostics.push(diag);
                }
            }
            _ => {}
        }
    }

    // Helper methods for naming conventions
    fn is_snake_case(&self, s: &str) -> bool {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_lowercase() || c.is_numeric() || c == '_')
            && !s.contains("__")
    }

    fn is_pascal_case(&self, s: &str) -> bool {
        !s.is_empty()
            && s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && !s.contains('_')
    }

    fn is_screaming_snake_case(&self, s: &str) -> bool {
        !s.is_empty()
            && s.chars()
                .all(|c| c.is_uppercase() || c.is_numeric() || c == '_')
            && !s.contains("__")
    }

    fn to_snake_case(&self, s: &str) -> String {
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() {
                if i > 0 {
                    result.push('_');
                }
                result.extend(c.to_lowercase());
            } else {
                result.push(c);
            }
        }
        result
    }

    fn to_pascal_case(&self, s: &str) -> String {
        s.split('_')
            .map(|word| {
                let mut chars = word.chars();
                match chars.next() {
                    None => String::new(),
                    Some(first) => first.to_uppercase().chain(chars).collect(),
                }
            })
            .collect()
    }

    fn to_screaming_snake_case(&self, s: &str) -> String {
        self.to_snake_case(s).to_uppercase()
    }

    /// Check if lint is enabled
    fn is_lint_enabled(&self, lint: &dyn Lint, ctx: &LintContext) -> bool {
        let level = ctx
            .levels
            .get(lint.id())
            .copied()
            .unwrap_or(lint.default_level());
        level != LintLevel::Allow
    }

    /// Lint an item
    fn lint_item(&self, item: &Item, ctx: &mut LintContext) {
        // Run item-level checks
        for lint in &self.lints {
            if self.is_lint_enabled(lint.as_ref(), ctx) {
                lint.check_item(item, ctx);
            }
        }

        match item {
            Item::Function(f) => {
                ctx.current_function = Some(f.name.clone());
                ctx.enter_scope();

                // Declare parameters
                for param in &f.params {
                    if let Pattern::Binding { name, .. } = &param.pattern {
                        ctx.declare_var(
                            name,
                            f.span, // Use function span as param doesn't have its own span
                            Some(param.ty.clone()),
                            param.is_mut,
                        );
                    }
                }

                // Check body
                self.lint_block(&f.body, ctx);

                // Report unused variables
                let unused: Vec<(String, Span)> = ctx
                    .unused_vars()
                    .into_iter()
                    .map(|(name, span)| (name.to_string(), span))
                    .collect();
                for (name, span) in unused {
                    ctx.report_with_suggestion(
                        &rules::UnusedVariable,
                        format!("unused variable `{}`", name),
                        span,
                        Suggestion {
                            message: "prefix with underscore to indicate intentionally unused"
                                .into(),
                            edits: vec![Edit {
                                span: Span {
                                    start: span.start,
                                    end: span.start,
                                },
                                replacement: "_".into(),
                            }],
                            applicability: Applicability::MachineApplicable,
                        },
                    );
                }

                ctx.exit_scope();
                ctx.current_function = None;
            }
            Item::Struct(s) => {
                // Check struct fields
                for field in &s.fields {
                    for lint in &self.lints {
                        if self.is_lint_enabled(lint.as_ref(), ctx) {
                            lint.check_type(&field.ty, ctx);
                        }
                    }
                }
            }
            Item::Enum(e) => {
                // Check enum variants
                for variant in &e.variants {
                    match &variant.data {
                        VariantData::Unit => {}
                        VariantData::Tuple(types) => {
                            for ty in types {
                                for lint in &self.lints {
                                    if self.is_lint_enabled(lint.as_ref(), ctx) {
                                        lint.check_type(ty, ctx);
                                    }
                                }
                            }
                        }
                        VariantData::Struct(fields) => {
                            for field in fields {
                                for lint in &self.lints {
                                    if self.is_lint_enabled(lint.as_ref(), ctx) {
                                        lint.check_type(&field.ty, ctx);
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Item::Impl(i) => {
                for impl_item in &i.items {
                    self.lint_impl_item(impl_item, ctx);
                }
            }
            Item::Trait(t) => {
                for trait_item in &t.items {
                    self.lint_trait_item(trait_item, ctx);
                }
            }
            _ => {}
        }
    }

    /// Lint an impl item
    fn lint_impl_item(&self, impl_item: &ImplItem, ctx: &mut LintContext) {
        match impl_item {
            ImplItem::Fn(f) => {
                // Lint the function as if it were a top-level function
                self.lint_item(&Item::Function(f.clone()), ctx);
            }
            ImplItem::Type(_) => {
                // Type aliases don't need special linting
            }
        }
    }

    /// Lint a trait item
    fn lint_trait_item(&self, trait_item: &TraitItem, ctx: &mut LintContext) {
        match trait_item {
            TraitItem::Fn(f) => {
                // Trait functions can have bodies or not
                // For now, just check the signature
                ctx.current_function = Some(f.name.clone());
                ctx.enter_scope();

                for param in &f.params {
                    if let Pattern::Binding { name, .. } = &param.pattern {
                        ctx.declare_var(
                            name,
                            Span { start: 0, end: 0 }, // TraitFnDef doesn't have span
                            Some(param.ty.clone()),
                            param.is_mut,
                        );
                    }
                }

                if let Some(body) = &f.default_body {
                    self.lint_block(body, ctx);
                }

                ctx.exit_scope();
                ctx.current_function = None;
            }
            TraitItem::Type(_) => {
                // Type aliases don't need special linting
            }
        }
    }

    /// Lint a block
    fn lint_block(&self, block: &Block, ctx: &mut LintContext) {
        ctx.enter_scope();

        for stmt in &block.stmts {
            self.lint_stmt(stmt, ctx);
        }

        // Report unused variables in this scope
        let unused: Vec<(String, Span)> = ctx
            .unused_vars()
            .into_iter()
            .map(|(name, span)| (name.to_string(), span))
            .collect();
        for (name, span) in unused {
            ctx.report_with_suggestion(
                &rules::UnusedVariable,
                format!("unused variable `{}`", name),
                span,
                Suggestion {
                    message: "prefix with underscore to indicate intentionally unused".into(),
                    edits: vec![Edit {
                        span: Span {
                            start: span.start,
                            end: span.start,
                        },
                        replacement: "_".into(),
                    }],
                    applicability: Applicability::MachineApplicable,
                },
            );
        }

        ctx.exit_scope();
    }

    /// Lint a statement
    fn lint_stmt(&self, stmt: &Stmt, ctx: &mut LintContext) {
        // Run statement-level checks
        for lint in &self.lints {
            if self.is_lint_enabled(lint.as_ref(), ctx) {
                lint.check_stmt(stmt, ctx);
            }
        }

        match stmt {
            Stmt::Let {
                is_mut,
                pattern,
                ty,
                value,
            } => {
                // Extract name from pattern
                if let Pattern::Binding { name, .. } = pattern {
                    ctx.declare_var(
                        name,
                        Span { start: 0, end: 0 }, // Dummy span - pattern doesn't have span
                        ty.clone(),
                        *is_mut,
                    );
                }

                // Check initializer
                if let Some(init) = value {
                    self.lint_expr(init, ctx);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.lint_expr(expr, ctx);
            }
            Stmt::Assign { target, value, .. } => {
                self.lint_expr(target, ctx);
                self.lint_expr(value, ctx);
            }
            Stmt::Empty | Stmt::MacroInvocation(_) => {}
        }
    }

    /// Lint an expression
    fn lint_expr(&self, expr: &Expr, ctx: &mut LintContext) {
        // Run expression-level checks
        for lint in &self.lints {
            if self.is_lint_enabled(lint.as_ref(), ctx) {
                lint.check_expr(expr, ctx);
            }
        }

        match expr {
            Expr::Path { path, .. } => {
                // Use first segment of path as variable reference
                if !path.segments.is_empty() {
                    ctx.use_var(&path.segments[0]);
                }
            }
            Expr::Binary { left, right, .. } => {
                self.lint_expr(left, ctx);
                self.lint_expr(right, ctx);
            }
            Expr::Unary { expr: inner, .. } => {
                self.lint_expr(inner, ctx);
            }
            Expr::Call { callee, args, .. } => {
                self.lint_expr(callee, ctx);
                for arg in args {
                    self.lint_expr(arg, ctx);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.lint_expr(receiver, ctx);
                for arg in args {
                    self.lint_expr(arg, ctx);
                }
            }
            Expr::Field { base, .. } => {
                self.lint_expr(base, ctx);
            }
            Expr::Index { base, index, .. } => {
                self.lint_expr(base, ctx);
                self.lint_expr(index, ctx);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.lint_expr(condition, ctx);
                self.lint_block(then_branch, ctx);
                if let Some(else_expr) = else_branch {
                    self.lint_expr(else_expr, ctx);
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.lint_expr(scrutinee, ctx);
                for arm in arms {
                    // Check pattern
                    for lint in &self.lints {
                        if self.is_lint_enabled(lint.as_ref(), ctx) {
                            lint.check_pattern(&arm.pattern, ctx);
                        }
                    }
                    if let Some(guard) = &arm.guard {
                        self.lint_expr(guard, ctx);
                    }
                    self.lint_expr(&arm.body, ctx);
                }
            }
            Expr::Loop { body, .. } => {
                self.lint_block(body, ctx);
            }
            Expr::While {
                condition, body, ..
            } => {
                self.lint_expr(condition, ctx);
                self.lint_block(body, ctx);
            }
            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => {
                self.lint_expr(iter, ctx);
                for lint in &self.lints {
                    if self.is_lint_enabled(lint.as_ref(), ctx) {
                        lint.check_pattern(pattern, ctx);
                    }
                }
                self.lint_block(body, ctx);
            }
            Expr::Block { block, .. } => {
                self.lint_block(block, ctx);
            }
            Expr::Return { value, .. } => {
                if let Some(v) = value {
                    self.lint_expr(v, ctx);
                }
            }
            Expr::Break { value, .. } => {
                if let Some(v) = value {
                    self.lint_expr(v, ctx);
                }
            }
            Expr::Closure { body, .. } => {
                self.lint_expr(body, ctx);
            }
            Expr::StructLit { fields, .. } => {
                for (_name, expr) in fields {
                    self.lint_expr(expr, ctx);
                }
            }
            Expr::Array { elements, .. } => {
                for elem in elements {
                    self.lint_expr(elem, ctx);
                }
            }
            Expr::Tuple { elements, .. } => {
                for elem in elements {
                    self.lint_expr(elem, ctx);
                }
            }
            Expr::Cast { expr, .. } => {
                self.lint_expr(expr, ctx);
            }
            Expr::Perform { args, .. } => {
                for arg in args {
                    self.lint_expr(arg, ctx);
                }
            }
            Expr::Handle { expr, .. } => {
                self.lint_expr(expr, ctx);
            }
            _ => {}
        }
    }
}

impl Default for Linter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_context() {
        let mut ctx = LintContext::new("test.sio", "let x = 1;");

        ctx.enter_scope();
        ctx.declare_var("x", Span { start: 4, end: 5 }, None, false);

        assert!(!ctx.unused_vars().is_empty());

        ctx.use_var("x");
        assert!(ctx.unused_vars().is_empty());

        ctx.exit_scope();
    }

    #[test]
    fn test_linter_creation() {
        let linter = Linter::new();
        assert!(!linter.lints().is_empty());
    }
}
