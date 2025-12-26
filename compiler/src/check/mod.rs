//! Type checker for Sounio
//!
//! This module implements type checking and produces HIR from the AST.
//! It handles:
//! - Type inference (bidirectional)
//! - Name resolution
//! - Effect checking
//! - Ownership/borrow checking
//! - Unit checking
//! - Epistemic type constraints
//! - Semantic type compatibility

pub mod compatibility;
pub mod diagnostics;
pub mod epistemic;

#[cfg(test)]
mod extern_tests;

use crate::ast::*;
use crate::common::{NodeId, Span};
use crate::hir::*;
use crate::macro_system::token_tree::{Delimiter, TokenTree};
use crate::types::{self, Type, TypeVar, effects::EffectInference, units::UnitChecker};
use miette::Result;
use std::collections::HashMap;

/// Type check an AST and produce HIR
pub fn check(ast: &Ast) -> Result<Hir> {
    let mut checker = TypeChecker::new();
    checker.check_program(ast)
}

/// Type checker state
pub struct TypeChecker {
    /// Type environment (variable -> type)
    env: TypeEnv,
    /// Type definitions
    type_defs: HashMap<String, TypeDef>,
    /// Effect inference context
    effects: EffectInference,
    /// Unit checker
    units: UnitChecker,
    /// Fresh type variable counter
    next_type_var: u32,
    /// Type constraints for unification
    constraints: Vec<TypeConstraint>,
    /// Errors accumulated during checking
    errors: Vec<TypeError>,
    /// Ontology alignments: (type1, type2) -> distance
    /// Key is ordered tuple: (min(t1,t2), max(t1,t2)) for symmetric lookup
    alignments: HashMap<(String, String), f64>,
    /// Function-level compatibility thresholds from #[compat] annotations
    fn_thresholds: HashMap<String, f64>,
    /// Default compatibility threshold
    default_threshold: f64,
    /// Reference to the AST for span lookup
    ast: Option<std::sync::Arc<Ast>>,
    /// Current function being type-checked (for threshold lookup)
    current_fn: Option<String>,
    /// Declared ontology prefixes (from `ontology X from "..."` declarations)
    ontology_prefixes: std::collections::HashSet<String>,
    /// Used ontology prefixes (to detect unused imports)
    used_ontology_prefixes: std::collections::HashSet<String>,
    /// Warnings accumulated during checking
    warnings: Vec<String>,
}

/// Type environment with scopes and module awareness
#[derive(Default)]
pub struct TypeEnv {
    scopes: Vec<Scope>,
    /// Module-qualified bindings: (module_path, name) -> binding
    /// Used for resolving qualified paths like `math.sin`
    module_bindings: HashMap<(Vec<String>, String), TypeBinding>,
}

#[derive(Default)]
struct Scope {
    bindings: HashMap<String, TypeBinding>,
}

/// Binding in environment
#[derive(Clone)]
struct TypeBinding {
    ty: Type,
    mutable: bool,
    used: bool,
    /// The module this binding originated from (if any)
    source_module: Option<ModuleId>,
}

/// Type definition (struct, enum, type alias)
#[derive(Clone)]
enum TypeDef {
    Struct {
        fields: Vec<(String, Type)>,
        linear: bool,
        affine: bool,
        /// The module this type was defined in
        source_module: Option<ModuleId>,
    },
    Enum {
        variants: Vec<(String, Vec<Type>)>,
        linear: bool,
        affine: bool,
        /// The module this type was defined in
        source_module: Option<ModuleId>,
    },
    Alias(Type, Span, Option<ModuleId>),
}

/// Type constraint for unification
#[derive(Debug)]
struct TypeConstraint {
    expected: Type,
    actual: Type,
    span: Span,
}

/// Type error
#[derive(Debug, Clone)]
pub struct TypeError {
    pub message: String,
    pub span: Span,
    pub code: String,
}

/// Structured type check result with detailed errors and warnings
#[derive(Debug)]
pub struct TypeCheckResult {
    pub hir: Option<Hir>,
    pub errors: Vec<TypeError>,
    pub warnings: Vec<String>,
}

/// Type check an AST and return structured result with errors
pub fn check_with_errors(ast: &Ast) -> TypeCheckResult {
    let mut checker = TypeChecker::new();
    match checker.check_program_internal(ast) {
        Ok(hir) => TypeCheckResult {
            hir: Some(hir),
            errors: checker.errors,
            warnings: checker.warnings,
        },
        Err(_) => TypeCheckResult {
            hir: None,
            errors: checker.errors,
            warnings: checker.warnings,
        },
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        Self {
            env: TypeEnv::default(),
            type_defs: HashMap::new(),
            effects: EffectInference::new(),
            units: UnitChecker::new(),
            next_type_var: 0,
            constraints: Vec::new(),
            errors: Vec::new(),
            alignments: HashMap::new(),
            fn_thresholds: HashMap::new(),
            default_threshold: 0.15, // Default threshold for semantic compatibility
            ast: None,
            current_fn: None,
            ontology_prefixes: std::collections::HashSet::new(),
            used_ontology_prefixes: std::collections::HashSet::new(),
            warnings: Vec::new(),
        }
    }

    /// Generate a fresh type variable
    fn fresh_type_var(&mut self) -> Type {
        let var = TypeVar(self.next_type_var);
        self.next_type_var += 1;
        Type::Var(var)
    }

    /// Add a type constraint
    fn constrain(&mut self, expected: Type, actual: Type, span: Span) {
        self.constraints.push(TypeConstraint {
            expected,
            actual,
            span,
        });
    }

    /// Report a type error (default code E0308)
    fn error(&mut self, message: impl Into<String>, span: Span) {
        self.errors.push(TypeError {
            message: message.into(),
            span,
            code: "E0308".to_string(),
        });
    }

    /// Report a type error with a specific code
    fn error_with_code(&mut self, code: &str, message: impl Into<String>, span: Span) {
        self.errors.push(TypeError {
            message: message.into(),
            span,
            code: code.to_string(),
        });
    }

    /// Expand type aliases recursively
    fn expand_type_alias(&self, ty: &Type) -> Type {
        match ty {
            Type::Named { name, args } => {
                // Check if this is a type alias
                if let Some(TypeDef::Alias(alias_ty, _, _)) = self.type_defs.get(name) {
                    // Recursively expand the alias
                    self.expand_type_alias(alias_ty)
                } else {
                    // Not an alias, but expand args recursively
                    Type::Named {
                        name: name.clone(),
                        args: args.iter().map(|a| self.expand_type_alias(a)).collect(),
                    }
                }
            }
            Type::Array { element, size } => Type::Array {
                element: Box::new(self.expand_type_alias(element)),
                size: *size,
            },
            Type::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.expand_type_alias(e)).collect())
            }
            Type::Ref {
                mutable,
                lifetime,
                inner,
            } => Type::Ref {
                mutable: *mutable,
                lifetime: lifetime.clone(),
                inner: Box::new(self.expand_type_alias(inner)),
            },
            Type::Function {
                params,
                return_type,
                effects,
            } => Type::Function {
                params: params.iter().map(|p| self.expand_type_alias(p)).collect(),
                return_type: Box::new(self.expand_type_alias(return_type)),
                effects: effects.clone(),
            },
            // Primitive types don't need expansion
            _ => ty.clone(),
        }
    }

    /// Get human-readable display name for a type
    fn type_display_name(&self, ty: &Type) -> String {
        match ty {
            Type::Unit => "()".to_string(),
            Type::Bool => "bool".to_string(),
            Type::I8 => "i8".to_string(),
            Type::I16 => "i16".to_string(),
            Type::I32 => "i32".to_string(),
            Type::I64 => "i64".to_string(),
            Type::U8 => "u8".to_string(),
            Type::U16 => "u16".to_string(),
            Type::U32 => "u32".to_string(),
            Type::U64 => "u64".to_string(),
            Type::F32 => "f32".to_string(),
            Type::F64 => "f64".to_string(),
            Type::String => "string".to_string(),
            Type::Named { name, args } => {
                if args.is_empty() {
                    name.clone()
                } else {
                    format!(
                        "{}<{}>",
                        name,
                        args.iter()
                            .map(|a| self.type_display_name(a))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            }
            Type::Ontology { namespace, term } => format!("{}:{}", namespace, term),
            Type::Array { element, size } => {
                format!(
                    "[{}; {}]",
                    self.type_display_name(element),
                    size.unwrap_or(0)
                )
            }
            Type::Tuple(types) => {
                let inner: Vec<_> = types.iter().map(|t| self.type_display_name(t)).collect();
                format!("({})", inner.join(", "))
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                let param_strs: Vec<_> = params.iter().map(|t| self.type_display_name(t)).collect();
                format!(
                    "fn({}) -> {}",
                    param_strs.join(", "),
                    self.type_display_name(return_type)
                )
            }
            Type::Var(v) => format!("?T{}", v.0),
            _ => format!("{:?}", ty),
        }
    }

    /// Extract span from an AST expression using the AST's span map
    fn expr_span(&self, expr: &Expr, ast: &Ast) -> Span {
        let id = self.expr_id(expr);
        ast.node_spans.get(&id).copied().unwrap_or_else(Span::dummy)
    }

    /// Extract NodeId from an AST expression
    fn expr_id(&self, expr: &Expr) -> NodeId {
        match expr {
            Expr::Literal { id, .. } => *id,
            Expr::Path { id, .. } => *id,
            Expr::Binary { id, .. } => *id,
            Expr::Unary { id, .. } => *id,
            Expr::Call { id, .. } => *id,
            Expr::MethodCall { id, .. } => *id,
            Expr::Field { id, .. } => *id,
            Expr::TupleField { id, .. } => *id,
            Expr::Index { id, .. } => *id,
            Expr::Cast { id, .. } => *id,
            Expr::Block { id, .. } => *id,
            Expr::If { id, .. } => *id,
            Expr::Match { id, .. } => *id,
            Expr::Loop { id, .. } => *id,
            Expr::While { id, .. } => *id,
            Expr::For { id, .. } => *id,
            Expr::Return { id, .. } => *id,
            Expr::Break { id, .. } => *id,
            Expr::Continue { id } => *id,
            Expr::Closure { id, .. } => *id,
            Expr::Tuple { id, .. } => *id,
            Expr::Array { id, .. } => *id,
            Expr::Range { id, .. } => *id,
            Expr::StructLit { id, .. } => *id,
            Expr::Try { id, .. } => *id,
            Expr::Perform { id, .. } => *id,
            Expr::Handle { id, .. } => *id,
            Expr::Sample { id, .. } => *id,
            Expr::Await { id, .. } => *id,
            Expr::AsyncBlock { id, .. } => *id,
            Expr::AsyncClosure { id, .. } => *id,
            Expr::Spawn { id, .. } => *id,
            Expr::Select { id, .. } => *id,
            Expr::Join { id, .. } => *id,
            Expr::OntologyTerm { id, .. } => *id,
            Expr::MacroInvocation(_) => NodeId(0),
            Expr::Do { id, .. } => *id,
            Expr::Counterfactual { id, .. } => *id,
            Expr::KnowledgeExpr { id, .. } => *id,
            _ => NodeId(0),
        }
    }

    pub fn check_program(&mut self, ast: &Ast) -> Result<Hir> {
        self.check_program_internal(ast)
    }

    fn check_program_internal(&mut self, ast: &Ast) -> Result<Hir> {
        // Store AST reference for span lookups
        self.ast = Some(std::sync::Arc::new(ast.clone()));

        let mut items = Vec::new();
        let mut externs = Vec::new();

        // First pass: collect ontology prefixes, type definitions, and alignments
        for item in &ast.items {
            self.collect_ontology_prefix(item);
        }
        for item in &ast.items {
            self.collect_type_def(item);
            self.collect_alignment(item);
            self.collect_fn_threshold(item);
        }

        // Validate that all ontology types use declared prefixes
        self.check_undefined_ontology_prefixes();

        // Check for circular type definitions
        self.check_circular_types();

        // Check for infinite-size structs (direct recursion without indirection)
        self.check_infinite_size_types();

        // Second pass: register function signatures in environment
        self.env.push_scope();
        for item in &ast.items {
            if let Item::Function(f) = item {
                let params: Vec<Type> = f
                    .params
                    .iter()
                    .map(|p| self.lower_type_expr(&p.ty))
                    .collect();
                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|t| self.lower_type_expr(t))
                    .unwrap_or(Type::Unit);
                let fn_type = Type::Function {
                    params,
                    return_type: Box::new(return_type),
                    effects: types::EffectSet::new(),
                };
                self.env.bind(f.name.clone(), fn_type, false);
            }
            if let Item::Extern(extern_block) = item {
                for ext_item in &extern_block.items {
                    if let ExternItem::Fn(ext_fn) = ext_item {
                        let params: Vec<Type> = ext_fn
                            .params
                            .iter()
                            .map(|p| self.lower_type_expr(&p.ty))
                            .collect();
                        let return_type = ext_fn
                            .return_type
                            .as_ref()
                            .map(|t| self.lower_type_expr(t))
                            .unwrap_or(Type::Unit);

                        let fn_type = Type::Function {
                            params,
                            return_type: Box::new(return_type),
                            effects: types::EffectSet::new(),
                        };

                        // Bind using the D-visible name; codegen/linking uses `link_name` later.
                        self.env.bind(ext_fn.name.clone(), fn_type, false);
                    }
                }
            }
            // Register associated functions from impl blocks (e.g., String::new)
            if let Item::Impl(impl_def) = item {
                // Get the type name from target_type
                let type_name = match &impl_def.target_type {
                    TypeExpr::Named { path, .. } => path.to_string(),
                    _ => continue,
                };
                for impl_item in &impl_def.items {
                    if let ImplItem::Fn(f) = impl_item {
                        // Check if it's an associated function (no self parameter)
                        let is_associated = f.params.first().map_or(true, |p| {
                            !matches!(&p.pattern, Pattern::Binding { name, .. } if name == "self")
                        });
                        if is_associated {
                            let params: Vec<Type> = f
                                .params
                                .iter()
                                .map(|p| self.lower_type_expr(&p.ty))
                                .collect();
                            let return_type = f
                                .return_type
                                .as_ref()
                                .map(|t| self.lower_type_expr(t))
                                .unwrap_or(Type::Unit);
                            let fn_type = Type::Function {
                                params,
                                return_type: Box::new(return_type),
                                effects: types::EffectSet::new(),
                            };
                            // Register as TypeName::method_name
                            let qualified_name = format!("{}::{}", type_name, f.name);
                            self.env.bind(qualified_name, fn_type, false);
                        }
                    }
                }
            }
            // Register functions from nested modules
            if let Item::Module(m) = item {
                self.collect_module_functions(m);
            }
            // Register global let/const bindings
            if let Item::Global(g) = item {
                let name = self.pattern_name(&g.pattern);
                let ty =
                    g.ty.as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or_else(|| self.fresh_type_var());
                self.env.bind(name, ty, g.is_mut);
            }
        }

        // Third pass: type check items
        for item in &ast.items {
            if let Item::Extern(extern_block) = item {
                externs.push(self.lower_extern_block(extern_block)?);
                continue;
            }
            if let Some(hir_item) = self.check_item(item)? {
                items.push(hir_item);
            }
        }

        self.env.pop_scope();

        // Solve type constraints
        self.solve_constraints()?;

        // Check for unused ontology imports and generate warnings
        self.check_unused_imports();

        if !self.errors.is_empty() {
            let messages: Vec<_> = self.errors.iter().map(|e| e.message.clone()).collect();
            return Err(miette::miette!("Type errors:\n{}", messages.join("\n")));
        }

        Ok(Hir { items, externs })
    }

    fn lower_extern_block(&mut self, block: &ExternBlock) -> Result<HirExternBlock> {
        let mut functions = Vec::new();

        for item in &block.items {
            if let ExternItem::Fn(f) = item {
                let params: Vec<HirParam> = f
                    .params
                    .iter()
                    .map(|p| {
                        let ty = self.lower_type_expr(&p.ty);
                        HirParam {
                            id: p.id,
                            name: self.pattern_name(&p.pattern),
                            ty: self.type_to_hir(&ty),
                            is_mut: p.is_mut,
                        }
                    })
                    .collect();

                let return_type = f
                    .return_type
                    .as_ref()
                    .map(|t| {
                        let ty = self.lower_type_expr(t);
                        self.type_to_hir(&ty)
                    })
                    .unwrap_or(HirType::Unit);

                functions.push(HirExternFn {
                    id: f.id,
                    name: f.name.clone(),
                    params,
                    return_type,
                    is_variadic: f.is_variadic,
                    link_name: f.link_name.clone(),
                });
            }
        }

        Ok(HirExternBlock {
            id: block.id,
            abi: block.abi.clone(),
            functions,
        })
    }

    /// Check for unused ontology imports and add warnings
    fn check_unused_imports(&mut self) {
        for prefix in &self.ontology_prefixes {
            if !self.used_ontology_prefixes.contains(prefix) {
                self.warnings.push(format!(
                    "unused_import: ontology prefix `{}` is declared but never used",
                    prefix
                ));
            }
        }
    }

    /// Parse vec! macro arguments into expressions
    /// For vec![a, b, c], extract the comma-separated expressions
    /// The args structure is: [Token("vec"), Token("!"), Delimited(Bracket, [...])]
    /// or just: [Delimited(Bracket, [...])]
    fn parse_vec_macro_args(&self, args: &[TokenTree]) -> Vec<Expr> {
        use crate::lexer::TokenKind;

        // Find the bracketed content - skip any leading vec! tokens
        let bracket_content = self.find_bracket_content(args);

        if bracket_content.is_empty() {
            return Vec::new();
        }

        // Parse comma-separated expressions from bracket content
        let mut exprs = Vec::new();
        let mut current_tokens = Vec::new();

        for tt in bracket_content {
            match tt {
                TokenTree::Token(tok) if tok.token.kind == TokenKind::Comma => {
                    if !current_tokens.is_empty() {
                        if let Some(expr) = self.tokens_to_simple_expr(&current_tokens) {
                            exprs.push(expr);
                        }
                        current_tokens.clear();
                    }
                }
                _ => {
                    current_tokens.push(tt.clone());
                }
            }
        }

        // Handle last expression (no trailing comma)
        if !current_tokens.is_empty() {
            if let Some(expr) = self.tokens_to_simple_expr(&current_tokens) {
                exprs.push(expr);
            }
        }

        exprs
    }

    /// Find the bracket content in vec! macro args
    /// The parser already unwraps the bracket, so args ARE the content.
    /// This just returns args directly unless there's a wrapper Delimited.
    fn find_bracket_content<'a>(&self, args: &'a [TokenTree]) -> &'a [TokenTree] {
        // For vec![a, b, c], the parser gives us args = [Token(a), Token(,), Token(b), ...]
        // directly (already unwrapped from the bracket).
        //
        // For recursive calls with [Delimited(Bracket, inner)], we need to unwrap.
        if args.len() == 1 {
            if let TokenTree::Delimited(Delimiter::Bracket, inner, _) = &args[0] {
                return inner;
            }
        }

        // Otherwise, args are the direct content
        args
    }

    /// Convert a sequence of tokens to a simple expression (handles nested vec!)
    fn tokens_to_simple_expr(&self, tokens: &[TokenTree]) -> Option<Expr> {
        use crate::lexer::TokenKind;

        if tokens.is_empty() {
            return None;
        }

        // Check for nested vec! macro: [Token("vec"), Token("!"), Delimited(Bracket, ...)]
        if tokens.len() >= 3 {
            if let (TokenTree::Token(first), TokenTree::Token(second)) = (&tokens[0], &tokens[1]) {
                if first.token.kind == TokenKind::Ident
                    && first.token.text == "vec"
                    && second.token.kind == TokenKind::Bang
                {
                    // This is a nested vec! macro - recursively parse it
                    let nested_exprs = self.parse_vec_macro_args(&tokens[2..]);
                    return Some(Expr::Array {
                        id: NodeId::dummy(),
                        elements: nested_exprs,
                    });
                }
            }
        }

        // For single token, convert directly
        if tokens.len() == 1 {
            if let TokenTree::Token(tok) = &tokens[0] {
                return self.token_to_expr(&tok.token);
            }
            // Handle delimited group (nested array without vec!)
            if let TokenTree::Delimited(Delimiter::Bracket, inner, _) = &tokens[0] {
                let mut inner_exprs = Vec::new();
                let mut current = Vec::new();
                for tt in inner.iter() {
                    match tt {
                        TokenTree::Token(tok) if tok.token.kind == TokenKind::Comma => {
                            if !current.is_empty() {
                                if let Some(e) = self.tokens_to_simple_expr(&current) {
                                    inner_exprs.push(e);
                                }
                                current.clear();
                            }
                        }
                        _ => current.push(tt.clone()),
                    }
                }
                if !current.is_empty() {
                    if let Some(e) = self.tokens_to_simple_expr(&current) {
                        inner_exprs.push(e);
                    }
                }
                return Some(Expr::Array {
                    id: NodeId::dummy(),
                    elements: inner_exprs,
                });
            }
        }

        None
    }

    /// Convert a single token to an expression
    fn token_to_expr(&self, token: &crate::lexer::Token) -> Option<Expr> {
        use crate::lexer::TokenKind;

        match token.kind {
            TokenKind::IntLit => {
                let value = token.text.parse::<i64>().ok()?;
                Some(Expr::Literal {
                    id: NodeId::dummy(),
                    value: Literal::Int(value),
                })
            }
            TokenKind::FloatLit => {
                let value = token.text.parse::<f64>().ok()?;
                Some(Expr::Literal {
                    id: NodeId::dummy(),
                    value: Literal::Float(value),
                })
            }
            TokenKind::StringLit => {
                let text = token.text.clone();
                Some(Expr::Literal {
                    id: NodeId::dummy(),
                    value: Literal::String(text),
                })
            }
            TokenKind::True => Some(Expr::Literal {
                id: NodeId::dummy(),
                value: Literal::Bool(true),
            }),
            TokenKind::False => Some(Expr::Literal {
                id: NodeId::dummy(),
                value: Literal::Bool(false),
            }),
            TokenKind::Ident => {
                let name = token.text.clone();
                Some(Expr::Path {
                    id: NodeId::dummy(),
                    path: Path::simple(&name),
                })
            }
            _ => None,
        }
    }

    fn collect_type_def(&mut self, item: &Item) {
        match item {
            Item::Struct(s) => {
                let fields: Vec<_> = s
                    .fields
                    .iter()
                    .map(|f| (f.name.clone(), self.lower_type_expr(&f.ty)))
                    .collect();
                self.type_defs.insert(
                    s.name.clone(),
                    TypeDef::Struct {
                        fields,
                        linear: s.modifiers.linear,
                        affine: s.modifiers.affine,
                        source_module: None, // TODO: extract from struct's module context
                    },
                );
            }
            Item::Enum(e) => {
                let variants: Vec<_> = e
                    .variants
                    .iter()
                    .map(|v| {
                        let types = match &v.data {
                            VariantData::Unit => Vec::new(),
                            VariantData::Tuple(types) => {
                                types.iter().map(|t| self.lower_type_expr(t)).collect()
                            }
                            VariantData::Struct(fields) => {
                                fields.iter().map(|f| self.lower_type_expr(&f.ty)).collect()
                            }
                        };
                        (v.name.clone(), types)
                    })
                    .collect();
                self.type_defs.insert(
                    e.name.clone(),
                    TypeDef::Enum {
                        variants,
                        linear: e.modifiers.linear,
                        affine: e.modifiers.affine,
                        source_module: None, // TODO: extract from enum's module context
                    },
                );
            }
            Item::TypeAlias(t) => {
                let ty = self.lower_type_expr(&t.ty);
                self.type_defs
                    .insert(t.name.clone(), TypeDef::Alias(ty, t.span, None)); // TODO: extract module context
            }
            Item::Module(m) => {
                // Recursively collect type definitions from nested modules
                if let Some(ref items) = m.items {
                    for item in items {
                        self.collect_type_def(item);
                    }
                }
            }
            _ => {}
        }
    }

    /// Collect function signatures from a module (recursive)
    fn collect_module_functions(&mut self, m: &ModuleDef) {
        if let Some(ref items) = m.items {
            for item in items {
                if let Item::Function(f) = item {
                    let params: Vec<Type> = f
                        .params
                        .iter()
                        .map(|p| self.lower_type_expr(&p.ty))
                        .collect();
                    let return_type = f
                        .return_type
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or(Type::Unit);
                    let fn_type = Type::Function {
                        params,
                        return_type: Box::new(return_type),
                        effects: types::EffectSet::new(),
                    };
                    // Register with module-qualified name
                    let qualified_name = format!("{}::{}", m.name, f.name);
                    self.env
                        .bind(qualified_name.clone(), fn_type.clone(), false);
                    // Also register unqualified for now (within module scope)
                    self.env.bind(f.name.clone(), fn_type, false);
                }
                // Recursively handle nested modules
                if let Item::Module(nested) = item {
                    self.collect_module_functions(nested);
                }
            }
        }
    }

    /// Collect ontology alignment declarations
    fn collect_alignment(&mut self, item: &Item) {
        if let Item::AlignDecl(align) = item {
            // Create canonical key (ordered pair for symmetric lookup)
            let t1 = format!("{}:{}", align.type1.prefix, align.type1.term);
            let t2 = format!("{}:{}", align.type2.prefix, align.type2.term);
            let key = if t1 <= t2 { (t1, t2) } else { (t2, t1) };
            self.alignments.insert(key, align.distance);
        }
    }

    /// Collect function-level compatibility thresholds from #[compat] annotations
    fn collect_fn_threshold(&mut self, item: &Item) {
        if let Item::Function(f) = item {
            // Check for #[compat(threshold = X)] attribute
            for attr in &f.attributes {
                if attr.name == "compat" {
                    match &attr.args {
                        AttributeArgs::Named(pairs) => {
                            for (key, value) in pairs {
                                if key == "threshold" {
                                    if let AttributeValue::Float(threshold) = value {
                                        self.validate_and_insert_threshold(&f.name, *threshold);
                                    }
                                }
                            }
                        }
                        AttributeArgs::Value(AttributeValue::Float(threshold)) => {
                            // Simple form: #[compat(0.2)]
                            self.validate_and_insert_threshold(&f.name, *threshold);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Validate threshold is in valid range [0.0, 1.0] and insert
    fn validate_and_insert_threshold(&mut self, fn_name: &str, threshold: f64) {
        if threshold < 0.0 {
            self.error(
                format!(
                    "Invalid threshold {} for function `{}`: threshold cannot be negative",
                    threshold, fn_name
                ),
                Span::dummy(),
            );
        } else if threshold > 1.0 {
            self.error(
                format!(
                    "Invalid threshold {} for function `{}`: threshold must be between 0.0 and 1.0",
                    threshold, fn_name
                ),
                Span::dummy(),
            );
        } else {
            self.fn_thresholds.insert(fn_name.to_string(), threshold);
        }
    }

    /// Collect ontology prefix declarations and check for duplicates
    fn collect_ontology_prefix(&mut self, item: &Item) {
        if let Item::OntologyImport(ont) = item {
            // Check for duplicate prefix
            if self.ontology_prefixes.contains(&ont.prefix) {
                self.error(
                    format!(
                        "Duplicate ontology prefix `{}`. Each ontology prefix can only be declared once.",
                        ont.prefix
                    ),
                    Span::dummy(),
                );
            } else {
                self.ontology_prefixes.insert(ont.prefix.clone());
            }
        }
    }

    /// Check that all ontology types reference declared prefixes
    fn check_undefined_ontology_prefixes(&mut self) {
        for (name, def) in &self.type_defs.clone() {
            match def {
                TypeDef::Alias(ty, span, _) => {
                    self.check_type_for_undefined_ontology(ty, name, *span);
                }
                TypeDef::Struct { fields, .. } => {
                    for (field_name, field_ty) in fields {
                        self.check_type_for_undefined_ontology(
                            field_ty,
                            &format!("{}.{}", name, field_name),
                            Span::dummy(),
                        );
                    }
                }
                TypeDef::Enum { variants, .. } => {
                    for (variant_name, types) in variants {
                        for ty in types {
                            self.check_type_for_undefined_ontology(
                                ty,
                                &format!("{}::{}", name, variant_name),
                                Span::dummy(),
                            );
                        }
                    }
                }
            }
        }
    }

    /// Check a single type for undefined ontology prefixes
    fn check_type_for_undefined_ontology(&mut self, ty: &Type, context: &str, span: Span) {
        match ty {
            Type::Ontology { namespace, term } => {
                if !self.ontology_prefixes.contains(namespace) {
                    self.error_with_code(
                        "E0412",
                        format!(
                            "Undefined ontology prefix `{}` in type `{}:{}` (used in {}). Add `ontology {} from \"...\";` declaration.",
                            namespace, namespace, term, context, namespace
                        ),
                        span,
                    );
                }
            }
            Type::Named { args, .. } => {
                for arg in args {
                    self.check_type_for_undefined_ontology(arg, context, span);
                }
            }
            Type::Array { element, .. } => {
                self.check_type_for_undefined_ontology(element, context, span);
            }
            Type::Tuple(types) => {
                for t in types {
                    self.check_type_for_undefined_ontology(t, context, span);
                }
            }
            Type::Function {
                params,
                return_type,
                ..
            } => {
                for p in params {
                    self.check_type_for_undefined_ontology(p, context, span);
                }
                self.check_type_for_undefined_ontology(return_type, context, span);
            }
            Type::Ref { inner, .. } => {
                self.check_type_for_undefined_ontology(inner, context, span);
            }
            _ => {}
        }
    }

    /// Check for circular type alias definitions (e.g., type A = B; type B = A;)
    fn check_circular_types(&mut self) {
        use std::collections::HashSet;

        // For each type alias, check if following the chain leads back to itself
        for (name, def) in &self.type_defs.clone() {
            if let TypeDef::Alias(ty, _, _) = def {
                let mut visited = HashSet::new();
                visited.insert(name.clone());

                if self.type_creates_cycle(ty, &mut visited) {
                    self.error(
                        format!("Circular type definition detected: `{}` references itself through type aliases", name),
                        Span::dummy(),
                    );
                }
            }
        }
    }

    /// Helper to detect if a type creates a cycle through type aliases
    fn type_creates_cycle(
        &self,
        ty: &Type,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        match ty {
            Type::Named { name, .. } => {
                if visited.contains(name) {
                    return true;
                }
                if let Some(TypeDef::Alias(inner, _, _)) = self.type_defs.get(name) {
                    visited.insert(name.clone());
                    self.type_creates_cycle(inner, visited)
                } else {
                    false
                }
            }
            Type::Array { element, .. } => self.type_creates_cycle(element, visited),
            Type::Tuple(types) => types.iter().any(|t| self.type_creates_cycle(t, visited)),
            Type::Function {
                params,
                return_type,
                ..
            } => {
                params.iter().any(|t| self.type_creates_cycle(t, visited))
                    || self.type_creates_cycle(return_type, visited)
            }
            _ => false,
        }
    }

    /// Check for infinite-size types (structs that contain themselves without indirection)
    fn check_infinite_size_types(&mut self) {
        use std::collections::HashSet;

        for (name, def) in &self.type_defs.clone() {
            if let TypeDef::Struct { fields, .. } = def {
                let mut visited = HashSet::new();
                visited.insert(name.clone());

                for (field_name, field_ty) in fields {
                    if self.type_has_infinite_size(field_ty, &mut visited.clone()) {
                        self.error(
                            format!(
                                "Struct `{}` has infinite size: field `{}` contains `{}` without indirection (use Box, &, or Option<Box<...>>)",
                                name, field_name, name
                            ),
                            Span::dummy(),
                        );
                        break;
                    }
                }
            }
        }
    }

    /// Check if a type has infinite size (contains itself without indirection)
    fn type_has_infinite_size(
        &self,
        ty: &Type,
        visited: &mut std::collections::HashSet<String>,
    ) -> bool {
        match ty {
            Type::Named { name, .. } => {
                if visited.contains(name) {
                    return true;
                }

                if let Some(def) = self.type_defs.get(name) {
                    match def {
                        TypeDef::Struct { fields, .. } => {
                            visited.insert(name.clone());
                            fields
                                .iter()
                                .any(|(_, field_ty)| self.type_has_infinite_size(field_ty, visited))
                        }
                        TypeDef::Alias(inner, _, _) => {
                            visited.insert(name.clone());
                            self.type_has_infinite_size(inner, visited)
                        }
                        TypeDef::Enum { .. } => false, // Enums are sized by their largest variant
                    }
                } else {
                    false
                }
            }
            // References and pointers provide indirection - they break the cycle
            Type::Ref { .. } => false,
            // Box, Option<Box<T>>, etc. also provide indirection
            // For now, we assume any generic type provides indirection (conservative)
            Type::Array { element, .. } => self.type_has_infinite_size(element, visited),
            Type::Tuple(types) => types
                .iter()
                .any(|t| self.type_has_infinite_size(t, visited)),
            _ => false,
        }
    }

    /// Get semantic distance between two ontology types
    fn get_semantic_distance(&self, t1: &str, t2: &str) -> Option<f64> {
        if t1 == t2 {
            return Some(0.0);
        }
        let key = if t1 <= t2 {
            (t1.to_string(), t2.to_string())
        } else {
            (t2.to_string(), t1.to_string())
        };
        self.alignments.get(&key).copied()
    }

    /// Check if two ontology types are compatible within given threshold
    fn check_ontology_compatibility(
        &self,
        expected_ns: &str,
        expected_term: &str,
        found_ns: &str,
        found_term: &str,
        threshold: f64,
    ) -> Result<f64, String> {
        let expected = format!("{}:{}", expected_ns, expected_term);
        let found = format!("{}:{}", found_ns, found_term);

        if let Some(distance) = self.get_semantic_distance(&expected, &found) {
            if distance <= threshold {
                Ok(distance)
            } else {
                Err(format!(
                    "semantic distance {} exceeds threshold {} between {} and {}",
                    distance, threshold, expected, found
                ))
            }
        } else {
            // No alignment found - check if same ontology/term
            if expected_ns == found_ns && expected_term == found_term {
                Ok(0.0)
            } else if expected_ns == found_ns {
                // Same ontology, different term - assume related
                Ok(0.5) // Default distance for same-ontology terms
            } else {
                // Different ontologies with no alignment
                Err(format!(
                    "no alignment found between {} and {} (different ontologies require explicit align declaration)",
                    expected, found
                ))
            }
        }
    }

    /// Check type compatibility with semantic distance threshold
    fn check_type_compatibility_with_threshold(
        &mut self,
        expected: &HirType,
        found: &HirType,
        threshold: f64,
        span: Span,
    ) {
        // Check if both types are ontology types
        match (expected, found) {
            (
                HirType::Ontology {
                    namespace: exp_ns,
                    term: exp_term,
                },
                HirType::Ontology {
                    namespace: found_ns,
                    term: found_term,
                },
            ) => {
                match self
                    .check_ontology_compatibility(exp_ns, exp_term, found_ns, found_term, threshold)
                {
                    Ok(distance) => {
                        // Types are compatible within threshold
                        if distance > 0.0 {
                            // Could add a note about semantic coercion here
                        }
                    }
                    Err(msg) => {
                        self.error(msg, span);
                    }
                }
            }
            // For named types, check if they resolve to ontology types
            (
                HirType::Named { name: exp_name, .. },
                HirType::Named {
                    name: found_name, ..
                },
            ) => {
                // Look up if these are type aliases to ontology types
                if let (Some(TypeDef::Alias(exp_ty, _, _)), Some(TypeDef::Alias(found_ty, _, _))) =
                    (self.type_defs.get(exp_name), self.type_defs.get(found_name))
                {
                    if let (
                        Type::Ontology {
                            namespace: exp_ns,
                            term: exp_term,
                        },
                        Type::Ontology {
                            namespace: found_ns,
                            term: found_term,
                        },
                    ) = (exp_ty, found_ty)
                    {
                        match self.check_ontology_compatibility(
                            &exp_ns,
                            &exp_term,
                            &found_ns,
                            &found_term,
                            threshold,
                        ) {
                            Ok(_) => {}
                            Err(msg) => {
                                // Include type alias names in error message
                                let full_msg = format!(
                                    "type mismatch: expected `{}` ({}:{}), found `{}` ({}:{}): {}",
                                    exp_name,
                                    exp_ns,
                                    exp_term,
                                    found_name,
                                    found_ns,
                                    found_term,
                                    msg
                                );
                                self.error(full_msg, span);
                            }
                        }
                    }
                }
            }
            // For mixed cases (named + ontology), also check
            (
                HirType::Named { name, .. },
                HirType::Ontology {
                    namespace: found_ns,
                    term: found_term,
                },
            ) => {
                if let Some(TypeDef::Alias(
                    Type::Ontology {
                        namespace: exp_ns,
                        term: exp_term,
                    },
                    _,
                    _,
                )) = self.type_defs.get(name)
                {
                    match self.check_ontology_compatibility(
                        exp_ns, exp_term, found_ns, found_term, threshold,
                    ) {
                        Ok(_) => {}
                        Err(msg) => {
                            self.error(msg, span);
                        }
                    }
                }
            }
            (
                HirType::Ontology {
                    namespace: exp_ns,
                    term: exp_term,
                },
                HirType::Named { name, .. },
            ) => {
                if let Some(TypeDef::Alias(
                    Type::Ontology {
                        namespace: found_ns,
                        term: found_term,
                    },
                    _,
                    _,
                )) = self.type_defs.get(name)
                {
                    match self.check_ontology_compatibility(
                        exp_ns, exp_term, found_ns, found_term, threshold,
                    ) {
                        Ok(_) => {}
                        Err(msg) => {
                            self.error(msg, span);
                        }
                    }
                }
            }
            // Other types: no ontology checking needed
            _ => {}
        }
    }

    fn check_item(&mut self, item: &Item) -> Result<Option<HirItem>> {
        match item {
            Item::Function(f) => {
                let hir_fn = self.check_function(f)?;
                Ok(Some(HirItem::Function(hir_fn)))
            }
            Item::Struct(s) => {
                let hir_struct = self.check_struct(s)?;
                Ok(Some(HirItem::Struct(hir_struct)))
            }
            Item::Enum(e) => {
                let hir_enum = self.check_enum(e)?;
                Ok(Some(HirItem::Enum(hir_enum)))
            }
            Item::Effect(e) => {
                let hir_effect = self.check_effect_def(e)?;
                Ok(Some(HirItem::Effect(hir_effect)))
            }
            Item::Handler(h) => {
                let hir_handler = self.check_handler_def(h)?;
                Ok(Some(HirItem::Handler(hir_handler)))
            }
            Item::Global(g) => {
                let hir_global = self.check_global(g)?;
                Ok(Some(HirItem::Global(hir_global)))
            }
            Item::Module(m) => {
                // Type check inline module items recursively
                // Items are flattened into the parent HIR for now
                if let Some(ref items) = m.items {
                    for item in items {
                        // Module items are checked but not collected here
                        // They'll be collected via the main item loop
                        let _ = self.check_item(item)?;
                    }
                }
                Ok(None)
            }
            Item::Import(_) => {
                // Imports are handled during name resolution
                // No HIR items produced
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn check_function(&mut self, f: &FnDef) -> Result<HirFn> {
        // Set current function for threshold lookup
        self.current_fn = Some(f.name.clone());

        self.env.push_scope();

        // Process parameters
        let mut params = Vec::new();
        for param in &f.params {
            let ty = self.lower_type_expr(&param.ty);
            let hir_ty = self.type_to_hir(&ty);

            // Bind parameter in environment
            if let Pattern::Binding { name, .. } = &param.pattern {
                self.env.bind(name.clone(), ty.clone(), param.is_mut);
            }

            params.push(HirParam {
                id: param.id,
                name: self.pattern_name(&param.pattern),
                ty: hir_ty,
                is_mut: param.is_mut,
            });
        }

        // Process return type
        let return_type = f
            .return_type
            .as_ref()
            .map(|t| self.lower_type_expr(t))
            .unwrap_or(Type::Unit);

        // Check body
        let body = self.check_block(&f.body, Some(&return_type))?;

        self.env.pop_scope();

        // Clear current function
        self.current_fn = None;

        // Determine ABI: use explicit ABI if specified, otherwise Rust
        let abi = f.modifiers.abi.clone().unwrap_or(crate::ast::Abi::Rust);

        // Function is exported if it's public AND has C ABI (for FFI)
        // or if it's just public (for D-to-D linking)
        let is_exported = matches!(f.visibility, crate::ast::Visibility::Public)
            || matches!(
                abi,
                crate::ast::Abi::C | crate::ast::Abi::CUnwind | crate::ast::Abi::System
            );

        Ok(HirFn {
            id: f.id,
            name: f.name.clone(),
            ty: HirFnType {
                params: params.clone(),
                return_type: Box::new(self.type_to_hir(&return_type)),
                effects: f
                    .effects
                    .iter()
                    .map(|e| HirEffect {
                        id: e.id,
                        name: e.name.to_string(),
                        operations: Vec::new(), // Operations are defined in effect declarations
                    })
                    .collect(),
            },
            body,
            abi,
            is_exported,
        })
    }

    fn check_struct(&mut self, s: &StructDef) -> Result<HirStruct> {
        let fields: Vec<_> = s
            .fields
            .iter()
            .map(|f| {
                let ty = self.lower_type_expr(&f.ty);
                HirField {
                    id: f.id,
                    name: f.name.clone(),
                    ty: self.type_to_hir(&ty),
                }
            })
            .collect();

        Ok(HirStruct {
            id: s.id,
            name: s.name.clone(),
            fields,
            is_linear: s.modifiers.linear,
            is_affine: s.modifiers.affine,
        })
    }

    fn check_enum(&mut self, e: &EnumDef) -> Result<HirEnum> {
        let variants: Vec<_> = e
            .variants
            .iter()
            .map(|v| {
                let fields = match &v.data {
                    VariantData::Unit => Vec::new(),
                    VariantData::Tuple(types) => {
                        let lowered: Vec<_> =
                            types.iter().map(|t| self.lower_type_expr(t)).collect();
                        lowered.iter().map(|t| self.type_to_hir(t)).collect()
                    }
                    VariantData::Struct(fields) => {
                        let lowered: Vec<_> =
                            fields.iter().map(|f| self.lower_type_expr(&f.ty)).collect();
                        lowered.iter().map(|t| self.type_to_hir(t)).collect()
                    }
                };
                HirVariant {
                    id: v.id,
                    name: v.name.clone(),
                    fields,
                }
            })
            .collect();

        Ok(HirEnum {
            id: e.id,
            name: e.name.clone(),
            variants,
            is_linear: e.modifiers.linear,
            is_affine: e.modifiers.affine,
        })
    }

    fn check_effect_def(&mut self, e: &EffectDef) -> Result<HirEffect> {
        let operations: Vec<_> = e
            .operations
            .iter()
            .map(|op| {
                let lowered_params: Vec<_> = op
                    .params
                    .iter()
                    .map(|p| self.lower_type_expr(&p.ty))
                    .collect();
                let params: Vec<_> = lowered_params.iter().map(|t| self.type_to_hir(t)).collect();
                let return_type = if let Some(t) = op.return_type.as_ref() {
                    let lowered = self.lower_type_expr(t);
                    self.type_to_hir(&lowered)
                } else {
                    HirType::Unit
                };

                HirEffectOp {
                    id: op.id,
                    name: op.name.clone(),
                    params,
                    return_type,
                }
            })
            .collect();

        Ok(HirEffect {
            id: e.id,
            name: e.name.clone(),
            operations,
        })
    }

    fn check_handler_def(&mut self, h: &HandlerDef) -> Result<HirHandler> {
        let cases: Vec<_> = h
            .cases
            .iter()
            .map(|case| {
                let params: Vec<_> = case
                    .params
                    .iter()
                    .map(|p| self.pattern_name(&p.pattern))
                    .collect();

                // Check handler case body expression
                let body = self
                    .check_expr(&case.body, None)
                    .unwrap_or_else(|_| HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Literal(HirLiteral::Unit),
                        ty: HirType::Unit,
                    });
                HirHandlerCase {
                    id: case.id,
                    op_name: case.name.clone(),
                    params,
                    body,
                }
            })
            .collect();

        Ok(HirHandler {
            id: h.id,
            name: h.name.clone(),
            effect: h.effect.to_string(),
            cases,
        })
    }

    fn check_global(&mut self, g: &GlobalDef) -> Result<HirGlobal> {
        let ty =
            g.ty.as_ref()
                .map(|t| self.lower_type_expr(t))
                .unwrap_or_else(|| self.fresh_type_var());

        // Check global value expression with expected type
        let expected_ty = ty.clone();
        let hir_ty = self.type_to_hir(&ty);
        let value = self
            .check_expr(&g.value, Some(&expected_ty))
            .unwrap_or_else(|_| HirExpr {
                id: NodeId::dummy(),
                kind: HirExprKind::Literal(HirLiteral::Unit),
                ty: hir_ty,
            });

        Ok(HirGlobal {
            id: g.id,
            name: self.pattern_name(&g.pattern),
            ty: self.type_to_hir(&ty),
            value,
            is_const: g.is_const,
        })
    }

    fn check_block(&mut self, block: &Block, expected: Option<&Type>) -> Result<HirBlock> {
        self.env.push_scope();

        let mut stmts = Vec::new();
        let mut result_ty = Type::Unit;

        for (i, stmt) in block.stmts.iter().enumerate() {
            let is_last = i == block.stmts.len() - 1;

            match stmt {
                Stmt::Let {
                    is_mut,
                    pattern,
                    ty,
                    value,
                } => {
                    // Check if we have an explicit type annotation
                    let has_annotation = ty.is_some();
                    let declared_ty = ty
                        .as_ref()
                        .map(|t| self.lower_type_expr(t))
                        .unwrap_or_else(|| self.fresh_type_var());

                    // Expand type aliases before type checking (e.g., A -> Vec<Vec<...>>)
                    let expanded_ty = self.expand_type_alias(&declared_ty);

                    let value_expr = value
                        .as_ref()
                        .map(|v| self.check_expr(v, Some(&expanded_ty)))
                        .transpose()?;

                    // Determine the final binding type:
                    // - If there's an explicit annotation, use the declared type
                    // - If no annotation, infer from the value expression's type
                    let binding_ty = if has_annotation {
                        declared_ty.clone()
                    } else if let Some(ref v_expr) = value_expr {
                        // Infer type from value expression
                        self.hir_type_to_type(&v_expr.ty)
                    } else {
                        declared_ty.clone()
                    };

                    // CRITICAL: Verify type compatibility between declared type and value type
                    if let Some(ref v_expr) = value_expr {
                        let actual_ty = self.hir_type_to_type(&v_expr.ty);

                        // Get span from the original AST value expression
                        let value_span = if let (Some(v), Some(ast_ref)) = (value, &self.ast) {
                            self.expr_span(v, ast_ref.as_ref())
                        } else {
                            Span::dummy()
                        };

                        // Get threshold for current function (from #[compat] annotation or default)
                        let threshold = self
                            .current_fn
                            .as_ref()
                            .and_then(|name| self.fn_thresholds.get(name).copied())
                            .unwrap_or(self.default_threshold);

                        // First check structural compatibility (use expanded type for comparison)
                        // Only check if we have an explicit annotation (otherwise we're inferring)
                        if has_annotation && !self.types_compatible(&expanded_ty, &actual_ty) {
                            let decl_name = self.type_display_name(&expanded_ty);
                            let actual_name = self.type_display_name(&actual_ty);
                            self.error(
                                format!(
                                    "Type mismatch: expected `{}`, found `{}`",
                                    decl_name, actual_name
                                ),
                                value_span,
                            );
                        }

                        // Also check semantic/ontology type compatibility with threshold
                        let declared_hir = self.type_to_hir(&expanded_ty);
                        self.check_type_compatibility_with_threshold(
                            &declared_hir,
                            &v_expr.ty,
                            threshold,
                            value_span,
                        );
                    }

                    if let Pattern::Binding { name, .. } = pattern {
                        self.env.bind(name.clone(), binding_ty.clone(), *is_mut);
                    }

                    stmts.push(HirStmt::Let {
                        name: self.pattern_name(pattern),
                        ty: self.type_to_hir(&binding_ty),
                        value: value_expr,
                        is_mut: *is_mut,
                        layout_hint: None, // Layout hints are filled in by layout synthesis pass
                    });
                }
                Stmt::Expr { expr, has_semi } => {
                    // Pass expected type for last expression without semicolon (implicit return)
                    let expr_expected = if is_last && !has_semi { expected } else { None };
                    let expr_result = self.check_expr(expr, expr_expected)?;

                    if is_last && !has_semi {
                        result_ty = self.hir_type_to_type(&expr_result.ty);
                    }

                    stmts.push(HirStmt::Expr(expr_result));
                }
                Stmt::Assign { target, op, value } => {
                    let target_expr = self.check_expr(target, None)?;
                    let value_expr =
                        self.check_expr(value, Some(&self.hir_type_to_type(&target_expr.ty)))?;

                    stmts.push(HirStmt::Assign {
                        target: target_expr,
                        value: value_expr,
                    });
                }
                Stmt::Empty | Stmt::MacroInvocation(_) => {}
            }
        }

        if let Some(exp) = expected {
            self.constrain(exp.clone(), result_ty.clone(), Span::dummy());
        }

        self.env.pop_scope();

        Ok(HirBlock {
            stmts,
            ty: self.type_to_hir(&result_ty),
        })
    }

    fn check_expr(&mut self, expr: &Expr, expected: Option<&Type>) -> Result<HirExpr> {
        let (kind, ty) = match expr {
            Expr::Literal { id, value } => {
                let (lit, ty) = self.check_literal_with_expected(value, expected);
                (HirExprKind::Literal(lit), ty)
            }

            Expr::Path { id, path } => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    if let Some(binding) = self.env.lookup(name) {
                        let ty = binding.ty.clone();
                        (HirExprKind::Local(name.clone()), self.type_to_hir(&ty))
                    } else if self.is_builtin_function(name) {
                        // Builtin function - return a function type
                        let builtin_ty = self.get_builtin_type(name);
                        (HirExprKind::Global(name.clone()), builtin_ty)
                    } else if self.is_builtin_variant(name) {
                        // Builtin enum variant (None, Some, Ok, Err)
                        let variant_ty = self.get_builtin_variant_type(name, expected);
                        (HirExprKind::Global(name.clone()), variant_ty)
                    } else {
                        self.error(format!("Unknown variable: {}", name), Span::dummy());
                        (HirExprKind::Local(name.clone()), HirType::Error)
                    }
                } else {
                    // Qualified path - try module-qualified lookup first
                    if let Some(binding) = self.env.lookup_qualified(&path.segments) {
                        let ty = binding.ty.clone();
                        let full_path = path.to_string();
                        (HirExprKind::Global(full_path), self.type_to_hir(&ty))
                    } else {
                        // Check if it's an enum variant (EnumName::Variant)
                        let type_name = &path.segments[0];
                        if let Some(TypeDef::Enum { variants, .. }) = self.type_defs.get(type_name)
                        {
                            if path.segments.len() == 2 {
                                let variant_name = &path.segments[1];
                                if let Some((_, variant_types)) =
                                    variants.iter().find(|(n, _)| n == variant_name)
                                {
                                    // Found enum variant
                                    let result_ty = HirType::Named {
                                        name: type_name.clone(),
                                        args: vec![],
                                    };
                                    (HirExprKind::Global(path.to_string()), result_ty)
                                } else {
                                    self.error(
                                        format!(
                                            "Unknown variant `{}` in enum `{}`",
                                            variant_name, type_name
                                        ),
                                        Span::dummy(),
                                    );
                                    (HirExprKind::Global(path.to_string()), HirType::Error)
                                }
                            } else {
                                (HirExprKind::Global(path.to_string()), HirType::Error)
                            }
                        } else {
                            // Module-qualified path not found - include module info in error if available
                            let error_msg = if let Some(ref resolved) = path.resolved_module {
                                format!(
                                    "Unknown qualified path `{}` (resolved to module {:?})",
                                    path.to_string(),
                                    resolved.path
                                )
                            } else {
                                format!("Unknown qualified path `{}`", path.to_string())
                            };
                            self.error(error_msg, Span::dummy());
                            (HirExprKind::Global(path.to_string()), HirType::Error)
                        }
                    }
                }
            }

            Expr::Binary {
                id,
                op,
                left,
                right,
            } => {
                // Iteratively flatten left-associative binary chains to avoid stack overflow
                // Collect chain: [(op, right_expr), ...] from innermost to outermost
                let mut chain: Vec<(BinaryOp, &Expr)> = Vec::new();
                let mut current = expr;

                // Walk down the left spine collecting operators and right operands
                while let Expr::Binary {
                    op: curr_op,
                    left: curr_left,
                    right: curr_right,
                    ..
                } = current
                {
                    chain.push((*curr_op, curr_right.as_ref()));
                    current = curr_left.as_ref();
                }

                // Now 'current' is the leftmost non-binary expression
                // Check it first
                let mut result = self.check_expr(current, None)?;

                // Process the chain in reverse (innermost to outermost)
                for (chain_op, chain_right) in chain.into_iter().rev() {
                    let right_expr =
                        self.check_expr(chain_right, Some(&self.hir_type_to_type(&result.ty)))?;
                    let result_ty = self.check_binary_units(chain_op, &result.ty, &right_expr.ty);
                    let hir_op = self.lower_binary_op(chain_op);

                    result = HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Binary {
                            op: hir_op,
                            left: Box::new(result),
                            right: Box::new(right_expr),
                        },
                        ty: result_ty,
                    };
                }

                (result.kind, result.ty)
            }

            Expr::Unary {
                id,
                op,
                expr: inner,
            } => {
                let inner_expr = self.check_expr(inner, None)?;
                let result_ty = self.unary_result_type(*op, &inner_expr.ty);
                let hir_op = self.lower_unary_op(*op);

                (
                    HirExprKind::Unary {
                        op: hir_op,
                        expr: Box::new(inner_expr),
                    },
                    result_ty,
                )
            }

            Expr::Call { id, callee, args } => {
                // Check if this is a method call disguised as Call(Field(...))
                if let Expr::Field { base, field, .. } = callee.as_ref() {
                    // This is a method call: base.field(args)
                    let receiver_expr = self.check_expr(base, None)?;
                    let receiver_ty = receiver_expr.ty.clone();

                    let arg_exprs: Vec<_> = args
                        .iter()
                        .map(|a| self.check_expr(a, None))
                        .collect::<Result<_>>()?;

                    let result_ty = self.get_method_return_type(&receiver_ty, field, &arg_exprs);

                    return Ok(HirExpr {
                        id: *id,
                        kind: HirExprKind::MethodCall {
                            receiver: Box::new(receiver_expr),
                            method: field.clone(),
                            args: arg_exprs,
                        },
                        ty: result_ty,
                    });
                }

                let callee_expr = self.check_expr(callee, None)?;
                // If we know the callee's parameter types, use them as expected types for args.
                // This enables context-driven literal typing (e.g., `1` -> `u8` when calling `fn(_, _, u8)`).
                let expected_param_tys: Option<Vec<Type>> = match &callee_expr.ty {
                    HirType::Fn { params, .. } => {
                        Some(params.iter().map(|p| self.hir_type_to_type(p)).collect())
                    }
                    _ => None,
                };

                let checked_args: Vec<_> = if let Some(param_tys) = expected_param_tys {
                    args.iter()
                        .enumerate()
                        .map(|(i, a)| self.check_expr(a, param_tys.get(i)))
                        .collect::<Result<_>>()?
                } else {
                    args.iter()
                        .map(|a| self.check_expr(a, None))
                        .collect::<Result<_>>()?
                };

                // Extract function name for threshold lookup
                let fn_name = match callee.as_ref() {
                    Expr::Path { path, .. } => path.segments.last().cloned(),
                    _ => None,
                };

                // Get threshold for this function (from #[compat] annotation or default)
                let threshold = fn_name
                    .as_ref()
                    .and_then(|name| self.fn_thresholds.get(name).copied())
                    .unwrap_or(self.default_threshold);

                // Special handling for Option/Result constructors
                // These need type inference from their arguments
                let special_constructor_ty = match fn_name.as_deref() {
                    Some("Some") => {
                        // Some(value) -> Option<typeof(value)>
                        let inner_ty = checked_args
                            .first()
                            .map(|a| a.ty.clone())
                            .unwrap_or(HirType::Unit);
                        Some(HirType::Named {
                            name: "Option".to_string(),
                            args: vec![inner_ty],
                        })
                    }
                    Some("None") => {
                        // None -> Option<T> where T is inferred from context
                        // For now, use expected type if available
                        if let Some(Type::Named { name, args }) = expected {
                            if name == "Option" {
                                Some(HirType::Named {
                                    name: "Option".to_string(),
                                    args: args.iter().map(|t| self.type_to_hir(t)).collect(),
                                })
                            } else {
                                Some(HirType::Named {
                                    name: "Option".to_string(),
                                    args: vec![HirType::Unit],
                                })
                            }
                        } else {
                            Some(HirType::Named {
                                name: "Option".to_string(),
                                args: vec![HirType::Unit],
                            })
                        }
                    }
                    Some("Ok") => {
                        // Ok(value) -> Result<typeof(value), E>
                        let ok_ty = checked_args
                            .first()
                            .map(|a| a.ty.clone())
                            .unwrap_or(HirType::Unit);
                        // Try to get error type from context
                        let err_ty = if let Some(Type::Named { name, args }) = expected {
                            if name == "Result" && args.len() > 1 {
                                self.type_to_hir(&args[1])
                            } else {
                                HirType::Unit
                            }
                        } else {
                            HirType::Unit
                        };
                        Some(HirType::Named {
                            name: "Result".to_string(),
                            args: vec![ok_ty, err_ty],
                        })
                    }
                    Some("Err") => {
                        // Err(value) -> Result<T, typeof(value)>
                        let err_ty = checked_args
                            .first()
                            .map(|a| a.ty.clone())
                            .unwrap_or(HirType::Unit);
                        // Try to get ok type from context
                        let ok_ty = if let Some(Type::Named { name, args }) = expected {
                            if name == "Result" && !args.is_empty() {
                                self.type_to_hir(&args[0])
                            } else {
                                HirType::Unit
                            }
                        } else {
                            HirType::Unit
                        };
                        Some(HirType::Named {
                            name: "Result".to_string(),
                            args: vec![ok_ty, err_ty],
                        })
                    }
                    _ => None,
                };

                // Extract return type and parameter types from function type
                let (result_ty, param_types) = if let Some(special_ty) = special_constructor_ty {
                    // Use the specially inferred type for constructors
                    (special_ty, vec![])
                } else {
                    match &callee_expr.ty {
                        HirType::Fn {
                            params,
                            return_type,
                            ..
                        } => (*return_type.clone(), params.clone()),
                        _ => (HirType::Unit, vec![]),
                    }
                };

                // Check ontological compatibility for each argument
                // We need to iterate with original args to get spans
                for (i, (checked_arg, param_ty)) in
                    checked_args.iter().zip(param_types.iter()).enumerate()
                {
                    // Get span from original AST argument
                    let arg_span = if let Some(ast_ref) = &self.ast {
                        args.get(i)
                            .map(|a| self.expr_span(a, ast_ref.as_ref()))
                            .unwrap_or_else(Span::dummy)
                    } else {
                        Span::dummy()
                    };

                    self.check_type_compatibility_with_threshold(
                        param_ty,
                        &checked_arg.ty,
                        threshold,
                        arg_span,
                    );
                }

                (
                    HirExprKind::Call {
                        func: Box::new(callee_expr),
                        args: checked_args,
                    },
                    result_ty,
                )
            }

            Expr::If {
                id,
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_expr = self.check_expr(condition, Some(&Type::Bool))?;
                let then_block = self.check_block(then_branch, expected)?;

                let else_expr = else_branch
                    .as_ref()
                    .map(|e| self.check_expr(e, expected))
                    .transpose()?;

                let result_ty = if else_expr.is_some() {
                    then_block.ty.clone()
                } else {
                    HirType::Unit
                };

                (
                    HirExprKind::If {
                        condition: Box::new(cond_expr),
                        then_branch: then_block,
                        else_branch: else_expr.map(Box::new),
                    },
                    result_ty,
                )
            }

            Expr::Block { id, block } => {
                let hir_block = self.check_block(block, expected)?;
                let ty = hir_block.ty.clone();
                (HirExprKind::Block(hir_block), ty)
            }

            Expr::Return { id, value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.check_expr(v, expected))
                    .transpose()?;

                // Return has Never type since control doesn't continue
                (HirExprKind::Return(val.map(Box::new)), HirType::Never)
            }

            Expr::Tuple { id, elements } => {
                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, None))
                    .collect::<Result<_>>()?;

                let tys: Vec<_> = exprs.iter().map(|e| e.ty.clone()).collect();
                let result_ty = HirType::Tuple(tys);

                (HirExprKind::Tuple(exprs), result_ty)
            }

            Expr::Array { id, elements } => {
                // Extract element type from expected type (either Array or Vec)
                let (elem_ty, is_vec) = expected
                    .and_then(|t| match t {
                        Type::Array { element, .. } => Some((element.as_ref().clone(), false)),
                        Type::Named { name, args } if name == "Vec" && args.len() == 1 => {
                            Some((args[0].clone(), true))
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| (self.fresh_type_var(), false));

                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, Some(&elem_ty)))
                    .collect::<Result<_>>()?;

                let elem_hir_ty = if exprs.is_empty() {
                    self.type_to_hir(&elem_ty)
                } else {
                    exprs[0].ty.clone()
                };

                // Return Vec<T> if expected type was Vec, otherwise Array
                let result_ty = if is_vec {
                    HirType::Named {
                        name: "Vec".to_string(),
                        args: vec![elem_hir_ty.clone()],
                    }
                } else {
                    HirType::Array {
                        element: Box::new(elem_hir_ty),
                        size: Some(exprs.len()),
                    }
                };

                (HirExprKind::Array(exprs), result_ty)
            }

            Expr::Range {
                id,
                start,
                end,
                inclusive,
            } => {
                // Type check start and end if present
                let start_expr = start
                    .as_ref()
                    .map(|e| self.check_expr(e, Some(&Type::I64)))
                    .transpose()?;
                let end_expr = end
                    .as_ref()
                    .map(|e| self.check_expr(e, Some(&Type::I64)))
                    .transpose()?;

                // Infer element type from start or end
                let elem_ty = start_expr
                    .as_ref()
                    .map(|e| e.ty.clone())
                    .or_else(|| end_expr.as_ref().map(|e| e.ty.clone()))
                    .unwrap_or(HirType::I64);

                // Range<T> type
                let range_ty = HirType::Named {
                    name: if *inclusive {
                        "RangeInclusive".to_string()
                    } else {
                        "Range".to_string()
                    },
                    args: vec![elem_ty],
                };

                (
                    HirExprKind::Range {
                        start: start_expr.map(Box::new),
                        end: end_expr.map(Box::new),
                        inclusive: *inclusive,
                    },
                    range_ty,
                )
            }

            Expr::Index { id, base, index } => {
                let base_expr = self.check_expr(base, None)?;
                let index_expr = self.check_expr(index, Some(&Type::I64))?;

                // Extract element type from indexable types
                let elem_ty = match &base_expr.ty {
                    HirType::Array { element, .. } => *element.clone(),
                    HirType::String => HirType::Char,
                    // Raw pointers are indexable - return inner type
                    HirType::RawPointer { inner, .. } => *inner.clone(),
                    // References to arrays
                    HirType::Ref { inner, .. } => {
                        if let HirType::Array { element, .. } = inner.as_ref() {
                            *element.clone()
                        } else {
                            HirType::Error
                        }
                    }
                    _ => HirType::Error,
                };

                (
                    HirExprKind::Index {
                        base: Box::new(base_expr),
                        index: Box::new(index_expr),
                    },
                    elem_ty,
                )
            }

            Expr::Field { id, base, field } => {
                let base_expr = self.check_expr(base, None)?;

                // Look up field type from struct definition
                let field_ty = if let HirType::Named { name, .. } = &base_expr.ty {
                    if let Some(TypeDef::Struct { fields, .. }) = self.type_defs.get(name) {
                        fields
                            .iter()
                            .find(|(n, _)| n == field)
                            .map(|(_, t)| self.type_to_hir(t))
                            .unwrap_or(HirType::Error)
                    } else {
                        HirType::Error
                    }
                } else {
                    HirType::Error
                };

                (
                    HirExprKind::Field {
                        base: Box::new(base_expr),
                        field: field.clone(),
                    },
                    field_ty,
                )
            }

            Expr::TupleField { id, base, index } => {
                let base_expr = self.check_expr(base, None)?;

                // Extract element type from tuple type
                let elem_ty = match &base_expr.ty {
                    HirType::Tuple(elements) => {
                        elements.get(*index).cloned().unwrap_or(HirType::Error)
                    }
                    _ => HirType::Error,
                };

                (
                    HirExprKind::TupleField {
                        base: Box::new(base_expr),
                        index: *index,
                    },
                    elem_ty,
                )
            }

            Expr::StructLit { id, path, fields } => {
                let struct_name = path.segments.last().cloned().unwrap_or_default();
                let checked_fields: Vec<_> = fields
                    .iter()
                    .map(|(name, expr)| {
                        let expr = self.check_expr(expr, None)?;
                        Ok((name.clone(), expr))
                    })
                    .collect::<Result<_>>()?;

                (
                    HirExprKind::Struct {
                        name: struct_name.clone(),
                        fields: checked_fields,
                    },
                    HirType::Named {
                        name: struct_name,
                        args: vec![],
                    },
                )
            }

            Expr::Loop { id, body } => {
                let body_block = self.check_block(body, None)?;
                (HirExprKind::Loop(body_block), HirType::Unit)
            }

            Expr::While {
                id: _,
                condition,
                body,
            } => {
                let cond_expr = self.check_expr(condition, Some(&Type::Bool))?;
                let body_block = self.check_block(body, None)?;

                // Use proper While HIR node - condition will be re-evaluated each iteration
                (
                    HirExprKind::While {
                        condition: Box::new(cond_expr),
                        body: body_block,
                    },
                    HirType::Unit,
                )
            }

            Expr::For {
                id: _,
                pattern,
                iter,
                body,
            } => {
                // Desugar: for i in start..end { body }
                // Into: { var __counter = start; while __counter < end { let i = __counter; body; __counter = __counter + 1 } }

                // Get the loop variable name from the pattern
                let loop_var = self.pattern_name(pattern);

                // Check if the iterator is a range expression
                match iter.as_ref() {
                    Expr::Range {
                        id: _,
                        start,
                        end,
                        inclusive,
                    } => {
                        // Type check start and end - let type inference determine the type
                        let start_expr = start
                            .as_ref()
                            .map(|e| self.check_expr(e, None))
                            .transpose()?
                            .unwrap_or_else(|| HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Literal(HirLiteral::Int(0)),
                                ty: HirType::I32,
                            });

                        let end_expr = end
                            .as_ref()
                            .map(|e| self.check_expr(e, None))
                            .transpose()?;

                        // Determine element type from start expression
                        let elem_ty = start_expr.ty.clone();
                        let is_inclusive = *inclusive;

                        // Generate unique counter variable name
                        let counter_var = format!("__for_counter_{}", self.next_type_var);
                        self.next_type_var += 1;

                        // Build the while condition: counter < end (or counter <= end for inclusive)
                        let cond_expr = if let Some(end_e) = end_expr {
                            HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Binary {
                                    op: if is_inclusive {
                                        HirBinaryOp::Le
                                    } else {
                                        HirBinaryOp::Lt
                                    },
                                    left: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(counter_var.clone()),
                                        ty: elem_ty.clone(),
                                    }),
                                    right: Box::new(end_e),
                                },
                                ty: HirType::Bool,
                            }
                        } else {
                            // Infinite range (1..) - always true
                            HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Literal(HirLiteral::Bool(true)),
                                ty: HirType::Bool,
                            }
                        };

                        // Push scope for the for loop body
                        self.env.push_scope();

                        // Define the loop variable in scope (immutable - it gets a new value each iteration)
                        self.env.bind(loop_var.clone(), self.hir_type_to_type(&elem_ty), false);

                        // Check the body
                        let body_block = self.check_block(body, None)?;

                        self.env.pop_scope();

                        // Build the loop body: let i = counter; <original body>; counter = counter + 1
                        let mut loop_stmts = Vec::new();

                        // let i = counter
                        loop_stmts.push(HirStmt::Let {
                            name: loop_var.clone(),
                            ty: elem_ty.clone(),
                            value: Some(HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Local(counter_var.clone()),
                                ty: elem_ty.clone(),
                            }),
                            is_mut: false,
                            layout_hint: None,
                        });

                        // Add original body statements
                        loop_stmts.extend(body_block.stmts);

                        // counter = counter + 1
                        loop_stmts.push(HirStmt::Assign {
                            target: HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Local(counter_var.clone()),
                                ty: elem_ty.clone(),
                            },
                            value: HirExpr {
                                id: NodeId::dummy(),
                                kind: HirExprKind::Binary {
                                    op: HirBinaryOp::Add,
                                    left: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Local(counter_var.clone()),
                                        ty: elem_ty.clone(),
                                    }),
                                    right: Box::new(HirExpr {
                                        id: NodeId::dummy(),
                                        kind: HirExprKind::Literal(HirLiteral::Int(1)),
                                        ty: elem_ty.clone(),
                                    }),
                                },
                                ty: elem_ty.clone(),
                            },
                        });

                        let while_body = HirBlock {
                            stmts: loop_stmts,
                            ty: HirType::Unit,
                        };

                        // Build the while loop
                        let while_expr = HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::While {
                                condition: Box::new(cond_expr),
                                body: while_body,
                            },
                            ty: HirType::Unit,
                        };

                        // Build the outer block: { var counter = start; while ... }
                        let outer_stmts = vec![
                            HirStmt::Let {
                                name: counter_var,
                                ty: elem_ty.clone(),
                                value: Some(start_expr),
                                is_mut: true,
                                layout_hint: None,
                            },
                            HirStmt::Expr(while_expr),
                        ];

                        (
                            HirExprKind::Block(HirBlock {
                                stmts: outer_stmts,
                                ty: HirType::Unit,
                            }),
                            HirType::Unit,
                        )
                    }
                    _ => {
                        // For now, only range-based for loops are supported
                        // TODO: Support iterators over arrays, slices, etc.
                        return Err(miette::miette!(
                            "for loops currently only support range expressions (e.g., for i in 1..10)"
                        ));
                    }
                }
            }

            Expr::Break { id, value } => {
                let val = value
                    .as_ref()
                    .map(|v| self.check_expr(v, None))
                    .transpose()?;
                (HirExprKind::Break(val.map(Box::new)), HirType::Never)
            }

            Expr::Continue { id } => (HirExprKind::Continue, HirType::Never),

            // ==================== EPISTEMIC EXPRESSIONS ====================

            // Do expression: do(X=1, Y=2) - list of interventions
            Expr::Do { id, interventions } => {
                // Lower each intervention as a sequence
                let mut do_exprs = Vec::new();
                for (var, val) in interventions {
                    let value_expr = self.check_expr(val, None)?;
                    do_exprs.push(HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Do {
                            variable: var.clone(),
                            value: Box::new(value_expr),
                        },
                        ty: HirType::Unit,
                    });
                }

                // If multiple interventions, wrap in a block
                if do_exprs.len() == 1 {
                    (do_exprs.pop().unwrap().kind, HirType::Unit)
                } else {
                    (
                        HirExprKind::Block(HirBlock {
                            stmts: do_exprs.into_iter().map(HirStmt::Expr).collect(),
                            ty: HirType::Unit,
                        }),
                        HirType::Unit,
                    )
                }
            }

            Expr::Counterfactual {
                id,
                factual,
                intervention,
                outcome,
            } => {
                let factual_expr = self.check_expr(factual, None)?;
                let intervention_expr = self.check_expr(intervention, None)?;
                let outcome_expr = self.check_expr(outcome, None)?;
                let outcome_ty = outcome_expr.ty.clone();

                (
                    HirExprKind::Counterfactual {
                        factual: Box::new(factual_expr),
                        intervention: Box::new(intervention_expr),
                        outcome: Box::new(outcome_expr),
                    },
                    outcome_ty,
                )
            }

            Expr::KnowledgeExpr {
                id,
                value,
                epsilon,
                validity,
                provenance,
            } => {
                let value_expr = self.check_expr(value, None)?;

                // Epsilon is optional
                let epsilon_expr = if let Some(eps) = epsilon {
                    self.check_expr(eps, Some(&Type::F64))?
                } else {
                    // Default epsilon of 1.0 (perfect confidence)
                    HirExpr {
                        id: NodeId::dummy(),
                        kind: HirExprKind::Literal(HirLiteral::Float(1.0)),
                        ty: HirType::F64,
                    }
                };

                let validity_expr = validity
                    .as_ref()
                    .map(|v| self.check_expr(v, None))
                    .transpose()?;

                let inner_ty = value_expr.ty.clone();
                let result_ty = HirType::Knowledge {
                    inner: Box::new(inner_ty),
                    epsilon_bound: None, // Could extract from epsilon if constant
                    provenance: None,
                };

                // Provenance is an expression, not a ProvenanceMarker - convert it
                let prov = provenance
                    .as_ref()
                    .map(|_| HirProvenance::Derived { sources: vec![] });

                (
                    HirExprKind::Knowledge {
                        value: Box::new(value_expr),
                        epsilon: Box::new(epsilon_expr),
                        validity: validity_expr.map(Box::new),
                        provenance: prov,
                    },
                    result_ty,
                )
            }

            Expr::Query {
                id,
                target,
                given,
                interventions,
            } => {
                let target_expr = self.check_expr(target, None)?;
                let given_exprs: Vec<_> = given
                    .iter()
                    .map(|g| self.check_expr(g, None))
                    .collect::<Result<_>>()?;

                // Interventions are (variable, value) pairs - lower each value
                let intervention_exprs: Vec<_> = interventions
                    .iter()
                    .map(|(var, val)| {
                        let val_expr = self.check_expr(val, None)?;
                        Ok(HirExpr {
                            id: NodeId::dummy(),
                            kind: HirExprKind::Do {
                                variable: var.clone(),
                                value: Box::new(val_expr),
                            },
                            ty: HirType::Unit,
                        })
                    })
                    .collect::<Result<_>>()?;

                // Query returns a probability (Knowledge[f64])
                let result_ty = HirType::Knowledge {
                    inner: Box::new(HirType::F64),
                    epsilon_bound: None,
                    provenance: None,
                };

                (
                    HirExprKind::Query {
                        target: Box::new(target_expr),
                        given: given_exprs,
                        interventions: intervention_exprs,
                    },
                    result_ty,
                )
            }

            // Observe expression: observe(data ~ distribution) for probabilistic programming
            Expr::Observe {
                id,
                data,
                distribution,
            } => {
                let data_expr = self.check_expr(data, None)?;
                let dist_expr = self.check_expr(distribution, None)?;

                // Create an observe expression - the variable is derived from the data expression
                let var_name = match &data_expr.kind {
                    HirExprKind::Local(name) => name.clone(),
                    _ => "_observed".to_string(),
                };

                (
                    HirExprKind::Observe {
                        variable: var_name,
                        value: Box::new(dist_expr),
                    },
                    HirType::Unit,
                )
            }

            // Uncertain expression: value with uncertainty (e.g., 5.0 ± 0.1)
            Expr::Uncertain {
                id,
                value,
                uncertainty,
            } => {
                let value_expr = self.check_expr(value, None)?;
                let uncertainty_expr = self.check_expr(uncertainty, None)?;
                let inner_ty = value_expr.ty.clone();

                // Convert uncertainty to epsilon (confidence bound)
                // For now, assume 2-sigma gives ~95% confidence
                let result_ty = HirType::Knowledge {
                    inner: Box::new(inner_ty),
                    epsilon_bound: Some(0.95),
                    provenance: None,
                };

                (
                    HirExprKind::Knowledge {
                        value: Box::new(value_expr),
                        epsilon: Box::new(uncertainty_expr), // Use uncertainty as epsilon proxy
                        validity: None,
                        provenance: Some(HirProvenance::Derived { sources: vec![] }),
                    },
                    result_ty,
                )
            }

            // Ontology term expression: prefix:term (e.g., chebi:aspirin, drugbank:DB00945)
            Expr::OntologyTerm {
                id: _,
                ontology,
                term,
            } => {
                // Track that this ontology prefix is used
                self.used_ontology_prefixes.insert(ontology.clone());

                let result_ty = HirType::Ontology {
                    namespace: ontology.clone(),
                    term: term.clone(),
                };

                (
                    HirExprKind::OntologyTerm {
                        namespace: ontology.clone(),
                        term: term.clone(),
                    },
                    result_ty,
                )
            }

            // Handle vec![] macro - treat as array literal with Vec type
            Expr::MacroInvocation(macro_inv) if macro_inv.name == "vec" => {
                // Parse vec! macro arguments as expressions
                // For vec![a, b, c], the args contain the comma-separated expressions
                let elements = self.parse_vec_macro_args(&macro_inv.args);

                // Determine element type from expected type or first element
                let (elem_ty, _is_vec) = expected
                    .and_then(|t| match t {
                        Type::Array { element, .. } => Some((element.as_ref().clone(), false)),
                        Type::Named { name, args } if name == "Vec" && args.len() == 1 => {
                            Some((args[0].clone(), true))
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| (self.fresh_type_var(), true)); // Default to Vec

                let exprs: Vec<_> = elements
                    .iter()
                    .map(|e| self.check_expr(e, Some(&elem_ty)))
                    .collect::<Result<_>>()?;

                let elem_hir_ty = if exprs.is_empty() {
                    self.type_to_hir(&elem_ty)
                } else {
                    exprs[0].ty.clone()
                };

                // vec! always produces Vec<T>
                let result_ty = HirType::Named {
                    name: "Vec".to_string(),
                    args: vec![elem_hir_ty],
                };

                (HirExprKind::Array(exprs), result_ty)
            }

            // Handle method calls (e.g., vec.is_empty(), vec.len(), etc.)
            Expr::MethodCall {
                id: _,
                receiver,
                method,
                args,
                ..
            } => {
                // First, check the receiver to get its type
                let receiver_expr = self.check_expr(receiver, None)?;
                let receiver_ty = receiver_expr.ty.clone();

                // Check arguments
                let arg_exprs: Vec<_> = args
                    .iter()
                    .map(|a| self.check_expr(a, None))
                    .collect::<Result<_>>()?;

                // Determine return type based on method name and receiver type
                let result_ty = self.get_method_return_type(&receiver_ty, method, &arg_exprs);

                (
                    HirExprKind::MethodCall {
                        receiver: Box::new(receiver_expr),
                        method: method.clone(),
                        args: arg_exprs,
                    },
                    result_ty,
                )
            }

            // Match expression handling
            Expr::Match {
                id: _,
                scrutinee,
                arms,
            } => {
                // Check the scrutinee expression
                let scrutinee_expr = self.check_expr(scrutinee, None)?;
                let scrutinee_ty = scrutinee_expr.ty.clone();

                // Check each arm
                let mut checked_arms = Vec::new();
                let mut arm_types = Vec::new();

                for arm in arms {
                    // Bind pattern variables based on scrutinee type
                    // For now, just check the body
                    let body_expr = self.check_expr(&arm.body, None)?;
                    arm_types.push(body_expr.ty.clone());

                    checked_arms.push(HirMatchArm {
                        pattern: self.lower_pattern(&arm.pattern),
                        guard: None,
                        body: body_expr,
                    });
                }

                // Determine the result type:
                // - If all arms return the same type, use that
                // - If one arm returns Never, use the other arm's type
                // - If arms differ and one is Unit (common with if-let without else), use Unit
                let result_ty = if arm_types.is_empty() {
                    HirType::Unit
                } else if arm_types.iter().all(|t| t == &arm_types[0]) {
                    arm_types[0].clone()
                } else {
                    // Check for Never type (for exhaustive patterns)
                    let non_never: Vec<_> =
                        arm_types.iter().filter(|t| **t != HirType::Never).collect();
                    if non_never.len() == 1 {
                        non_never[0].clone()
                    } else if arm_types.iter().any(|t| *t == HirType::Unit) {
                        // If any arm returns Unit (like if-let without else), the whole expression is Unit
                        HirType::Unit
                    } else {
                        // Default to first arm's type
                        arm_types[0].clone()
                    }
                };

                (
                    HirExprKind::Match {
                        scrutinee: Box::new(scrutinee_expr),
                        arms: checked_arms,
                    },
                    result_ty,
                )
            }

            // Cast expression: expr as Type
            Expr::Cast { id: _, expr, ty } => {
                // Check the inner expression
                let inner_expr = self.check_expr(expr, None)?;

                // Convert target type from TypeExpr to HirType
                let target_type = self.lower_type_expr(ty);
                let hir_target = self.type_to_hir(&target_type);

                (
                    HirExprKind::Cast {
                        expr: Box::new(inner_expr),
                        target: hir_target.clone(),
                    },
                    hir_target,
                )
            }

            // Simplified handling for other expressions
            _ => {
                // For now, return a placeholder
                (HirExprKind::Literal(HirLiteral::Unit), HirType::Unit)
            }
        };

        let id = match expr {
            Expr::Literal { id, .. }
            | Expr::Path { id, .. }
            | Expr::Binary { id, .. }
            | Expr::Unary { id, .. }
            | Expr::Call { id, .. }
            | Expr::If { id, .. }
            | Expr::Block { id, .. }
            | Expr::Return { id, .. }
            | Expr::Tuple { id, .. }
            | Expr::Array { id, .. }
            | Expr::Cast { id, .. }
            | Expr::OntologyTerm { id, .. } => *id,
            _ => NodeId::dummy(),
        };

        Ok(HirExpr { id, kind, ty })
    }

    fn check_literal_with_expected(
        &self,
        lit: &Literal,
        expected: Option<&Type>,
    ) -> (HirLiteral, HirType) {
        match lit {
            Literal::Unit => (HirLiteral::Unit, HirType::Unit),
            Literal::Bool(b) => (HirLiteral::Bool(*b), HirType::Bool),
            Literal::Int(i) => {
                // Infer integer type from context if available
                let ty = match expected {
                    Some(Type::I8) => HirType::I8,
                    Some(Type::I16) => HirType::I16,
                    Some(Type::I32) => HirType::I32,
                    Some(Type::I64) => HirType::I64,
                    Some(Type::I128) => HirType::I128,
                    Some(Type::Isize) => HirType::Isize,
                    Some(Type::U8) => HirType::U8,
                    Some(Type::U16) => HirType::U16,
                    Some(Type::U32) => HirType::U32,
                    Some(Type::U64) => HirType::U64,
                    Some(Type::U128) => HirType::U128,
                    Some(Type::Usize) => HirType::Usize,
                    Some(Type::F32) => HirType::F32,
                    Some(Type::F64) => HirType::F64,
                    _ => HirType::I64, // Default to i64
                };
                (HirLiteral::Int(*i), ty)
            }
            Literal::Float(f) => {
                // Infer float type from context if available
                let ty = match expected {
                    Some(Type::F32) => HirType::F32,
                    _ => HirType::F64, // Default to f64
                };
                (HirLiteral::Float(*f), ty)
            }
            Literal::Char(c) => (HirLiteral::Char(*c), HirType::Char),
            Literal::String(s) => (HirLiteral::String(s.clone()), HirType::String),
            // Unit literals: create Quantity type with unit information
            Literal::IntUnit(i, unit) => {
                let hir_unit = self.parse_unit_string(unit);
                (
                    HirLiteral::Int(*i),
                    HirType::Quantity {
                        numeric: Box::new(HirType::I64),
                        unit: hir_unit,
                    },
                )
            }
            Literal::FloatUnit(f, unit) => {
                let hir_unit = self.parse_unit_string(unit);
                (
                    HirLiteral::Float(*f),
                    HirType::Quantity {
                        numeric: Box::new(HirType::F64),
                        unit: hir_unit,
                    },
                )
            }
        }
    }

    /// Check unit compatibility for binary operations and compute result type
    fn check_binary_units(&mut self, op: BinaryOp, left: &HirType, right: &HirType) -> HirType {
        // Extract units from quantity types
        let (left_numeric, left_unit) = self.extract_quantity(left);
        let (right_numeric, right_unit) = self.extract_quantity(right);

        match op {
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::PlusMinus => {
                // Addition/subtraction requires compatible units
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        if !lu.is_compatible(ru) {
                            self.error(
                                format!(
                                    "Unit mismatch in {}: cannot {} {} and {}",
                                    if op == BinaryOp::Add {
                                        "addition"
                                    } else {
                                        "subtraction"
                                    },
                                    if op == BinaryOp::Add {
                                        "add"
                                    } else {
                                        "subtract"
                                    },
                                    lu.format(),
                                    ru.format()
                                ),
                                Span::dummy(),
                            );
                            return HirType::Error;
                        }
                        // Result has same unit as operands
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: lu.clone(),
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        self.error(
                            format!(
                                "Cannot {} values with and without units",
                                if op == BinaryOp::Add {
                                    "add"
                                } else {
                                    "subtract"
                                }
                            ),
                            Span::dummy(),
                        );
                        HirType::Error
                    }
                    (None, None) => left_numeric.clone(),
                }
            }
            BinaryOp::Mul => {
                // Multiplication: units multiply
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        let result_unit = lu.multiply(ru);
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: result_unit,
                        }
                    }
                    (Some(lu), None) => HirType::Quantity {
                        numeric: Box::new(left_numeric.clone()),
                        unit: lu.clone(),
                    },
                    (None, Some(ru)) => HirType::Quantity {
                        numeric: Box::new(left_numeric.clone()),
                        unit: ru.clone(),
                    },
                    (None, None) => left_numeric.clone(),
                }
            }
            BinaryOp::Div => {
                // Division: units divide
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        let result_unit = lu.divide(ru);
                        if result_unit.is_dimensionless() {
                            left_numeric.clone()
                        } else {
                            HirType::Quantity {
                                numeric: Box::new(left_numeric.clone()),
                                unit: result_unit,
                            }
                        }
                    }
                    (Some(lu), None) => HirType::Quantity {
                        numeric: Box::new(left_numeric.clone()),
                        unit: lu.clone(),
                    },
                    (None, Some(ru)) => {
                        // Dividing dimensionless by unit gives inverse unit
                        let result_unit = HirUnit::dimensionless().divide(ru);
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: result_unit,
                        }
                    }
                    (None, None) => left_numeric.clone(),
                }
            }
            BinaryOp::Rem => {
                // Remainder: same rules as division for compatibility, result has left's unit
                match (&left_unit, &right_unit) {
                    (Some(lu), Some(ru)) => {
                        if !lu.is_compatible(ru) {
                            self.error(
                                format!(
                                    "Unit mismatch in remainder: incompatible units {} and {}",
                                    lu.format(),
                                    ru.format()
                                ),
                                Span::dummy(),
                            );
                            return HirType::Error;
                        }
                        HirType::Quantity {
                            numeric: Box::new(left_numeric.clone()),
                            unit: lu.clone(),
                        }
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        self.error(
                            "Cannot compute remainder of values with and without units".to_string(),
                            Span::dummy(),
                        );
                        HirType::Error
                    }
                    (None, None) => left_numeric.clone(),
                }
            }
            // Comparison operators: units must be compatible, result is bool
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge => {
                if let (Some(lu), Some(ru)) = (&left_unit, &right_unit) {
                    if !lu.is_compatible(ru) {
                        self.error(
                            format!(
                                "Unit mismatch in comparison: cannot compare {} and {}",
                                lu.format(),
                                ru.format()
                            ),
                            Span::dummy(),
                        );
                    }
                }
                HirType::Bool
            }
            // Logical operators
            BinaryOp::And | BinaryOp::Or => HirType::Bool,
            // Bitwise operators: no unit handling
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => left.clone(),
            // Concatenation: combine array sizes
            BinaryOp::Concat => {
                match (left, right) {
                    (
                        HirType::Array {
                            element: left_elem,
                            size: left_size,
                        },
                        HirType::Array {
                            element: _right_elem,
                            size: right_size,
                        },
                    ) => {
                        // Combine sizes if both are known
                        let combined_size = match (left_size, right_size) {
                            (Some(l), Some(r)) => Some(l + r),
                            _ => None, // Unknown size if either is unknown
                        };
                        HirType::Array {
                            element: left_elem.clone(),
                            size: combined_size,
                        }
                    }
                    // For Vec or other types, just return a Vec
                    (HirType::Named { name, args }, _) if name == "Vec" => HirType::Named {
                        name: name.clone(),
                        args: args.clone(),
                    },
                    // Default: keep left type
                    _ => left.clone(),
                }
            }
        }
    }

    /// Extract the numeric type and optional unit from a type
    fn extract_quantity(&self, ty: &HirType) -> (HirType, Option<HirUnit>) {
        match ty {
            HirType::Quantity { numeric, unit } => (*numeric.clone(), Some(unit.clone())),
            _ => (ty.clone(), None),
        }
    }

    /// Check if a name is a builtin function
    fn is_builtin_function(&self, name: &str) -> bool {
        matches!(
            name,
            "print"
                | "println"
                | "assert"
                | "assert_eq"
                | "len"
                | "type_of"
                | "Some"
                | "None"
                | "Ok"
                | "Err"
                | "dbg"
                | "panic"
                | "format"
                | "read_line"
                | "parse_int"
                | "parse_float"
                | "to_string"
                | "sqrt"
                | "abs"
                | "sin"
                | "cos"
                | "tan"
                | "exp"
                | "log"
                | "pow"
                | "floor"
                | "ceil"
                | "round"
                | "min"
                | "max"
                // Linear algebra constructors
                | "vec2"
                | "vec3"
                | "vec4"
                | "mat2"
                | "mat3"
                | "mat4"
                | "quat"
                // Vector operations
                | "dot"
                | "cross"
                | "normalize"
                | "length"
                | "length_squared"
                // Quaternion operations
                | "quat_mul"
                | "quat_conj"
                | "quat_inv"
                | "quat_normalize"
                | "quat_identity"
                // Matrix operations
                | "mat_mul"
                | "transpose"
                | "inverse"
                | "determinant"
                // Interpolation
                | "lerp"
                | "slerp"
                // Conversions
                | "quat_to_euler"
                | "euler_to_quat"
                | "quat_to_mat3"
                | "quat_to_mat4"
                | "mat3_to_quat"
                // Quaternion Embeddings (Knowledge Graph) - arXiv:1904.10281
                | "hamilton_product"
                | "quat_rotate_vec"
                | "quat_score"
                | "quat_embed_init"
                | "quat_normalize_embed"
                | "quat_inner_product"
                // Automatic Differentiation
                | "dual"
                | "dual_value"
                | "dual_deriv"
                | "grad"
                | "jacobian"
                | "hessian"
                // FFI / Raw pointer operations
                | "null_ptr"
                | "null_mut"
                | "is_null"
                | "ptr_eq"
                | "ptr_addr"
                | "ptr_from_addr"
                | "ptr_from_addr_mut"
                | "ptr_offset"
                | "ptr_add"
                | "ptr_sub"
                | "ptr_diff"
                | "as_const"
                | "as_mut"
                | "size_of"
                | "align_of"
        )
    }

    /// Get the type of a builtin function
    fn get_builtin_type(&self, name: &str) -> HirType {
        // For simplicity, most builtins are treated as functions that take any args and return unit or the appropriate type
        match name {
            "print" | "println" | "dbg" | "panic" | "assert" | "assert_eq" => {
                // These return unit
                HirType::Fn {
                    params: vec![], // Variadic, but we'll be lenient
                    return_type: Box::new(HirType::Unit),
                }
            }
            "len" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::I64),
            },
            "type_of" | "format" | "to_string" | "read_line" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::String),
            },
            "parse_int" | "parse_float" => HirType::Fn {
                params: vec![HirType::String],
                return_type: Box::new(HirType::Unit), // Actually returns Option, simplified
            },
            "sqrt" | "abs" | "sin" | "cos" | "tan" | "exp" | "log" | "pow" | "floor" | "ceil"
            | "round" | "min" | "max" => HirType::Fn {
                params: vec![HirType::F64],
                return_type: Box::new(HirType::F64),
            },
            "Some" | "Ok" | "Err" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::Unit), // Generic, simplified
            },
            // None without context - will be handled by get_builtin_variant_type with expected
            "None" => HirType::Named {
                name: "Option".to_string(),
                args: vec![HirType::Unit],
            },
            // Linear algebra constructors
            "vec2" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Vec2),
            },
            "vec3" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Vec3),
            },
            "vec4" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32, HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Vec4),
            },
            "mat2" => HirType::Fn {
                params: vec![HirType::F32; 4], // 2x2 = 4 floats
                return_type: Box::new(HirType::Mat2),
            },
            "mat3" => HirType::Fn {
                params: vec![HirType::F32; 9], // 3x3 = 9 floats
                return_type: Box::new(HirType::Mat3),
            },
            "mat4" => HirType::Fn {
                params: vec![HirType::F32; 16], // 4x4 = 16 floats
                return_type: Box::new(HirType::Mat4),
            },
            "quat" => HirType::Fn {
                params: vec![HirType::F32, HirType::F32, HirType::F32, HirType::F32],
                return_type: Box::new(HirType::Quat),
            },
            // Vector operations
            "dot" => HirType::Fn {
                params: vec![HirType::Vec3, HirType::Vec3],
                return_type: Box::new(HirType::F32),
            },
            "cross" => HirType::Fn {
                params: vec![HirType::Vec3, HirType::Vec3],
                return_type: Box::new(HirType::Vec3),
            },
            "normalize" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::Vec3),
            },
            "length" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::F32),
            },
            "length_squared" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::F32),
            },
            // Quaternion operations
            "quat_mul" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_conj" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_inv" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_normalize" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            "quat_identity" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::Quat),
            },
            // Matrix operations
            "mat_mul" => HirType::Fn {
                params: vec![HirType::Mat4, HirType::Mat4],
                return_type: Box::new(HirType::Mat4),
            },
            "transpose" => HirType::Fn {
                params: vec![HirType::Mat4],
                return_type: Box::new(HirType::Mat4),
            },
            "inverse" => HirType::Fn {
                params: vec![HirType::Mat4],
                return_type: Box::new(HirType::Mat4),
            },
            "determinant" => HirType::Fn {
                params: vec![HirType::Mat4],
                return_type: Box::new(HirType::F32),
            },
            // Interpolation
            "lerp" => HirType::Fn {
                params: vec![HirType::Vec3, HirType::Vec3, HirType::F32],
                return_type: Box::new(HirType::Vec3),
            },
            "slerp" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat, HirType::F32],
                return_type: Box::new(HirType::Quat),
            },
            // Conversions
            "quat_to_euler" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Vec3),
            },
            "euler_to_quat" => HirType::Fn {
                params: vec![HirType::Vec3],
                return_type: Box::new(HirType::Quat),
            },
            "quat_to_mat3" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Mat3),
            },
            "quat_to_mat4" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Mat4),
            },
            "mat3_to_quat" => HirType::Fn {
                params: vec![HirType::Mat3],
                return_type: Box::new(HirType::Quat),
            },
            // Quaternion Embeddings (Knowledge Graph) - arXiv:1904.10281
            // Hamilton product: q1 ⊗ q2 - captures inter-dependencies between components
            "hamilton_product" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            // Rotate vector by quaternion: q * v * q^(-1)
            "quat_rotate_vec" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Vec3],
                return_type: Box::new(HirType::Vec3),
            },
            // Score triple (head, relation, tail) for knowledge graph completion
            // Returns scalar score: <h ⊗ r, t> where ⊗ is Hamilton product
            "quat_score" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::F32),
            },
            // Initialize quaternion embedding with random unit quaternion
            "quat_embed_init" => HirType::Fn {
                params: vec![HirType::I32], // seed
                return_type: Box::new(HirType::Quat),
            },
            // Normalize to unit quaternion for embeddings
            "quat_normalize_embed" => HirType::Fn {
                params: vec![HirType::Quat],
                return_type: Box::new(HirType::Quat),
            },
            // Inner product of two quaternion embeddings: sum of component-wise products
            "quat_inner_product" => HirType::Fn {
                params: vec![HirType::Quat, HirType::Quat],
                return_type: Box::new(HirType::F32),
            },

            // ==================== AUTOMATIC DIFFERENTIATION ====================
            // Dual number constructor: dual(value, derivative)
            "dual" => HirType::Fn {
                params: vec![HirType::F64, HirType::F64],
                return_type: Box::new(HirType::Dual),
            },
            // Extract value component from dual number
            "dual_value" => HirType::Fn {
                params: vec![HirType::Dual],
                return_type: Box::new(HirType::F64),
            },
            // Extract derivative component from dual number
            "dual_deriv" => HirType::Fn {
                params: vec![HirType::Dual],
                return_type: Box::new(HirType::F64),
            },
            // Compute gradient of a function at a point
            // grad(f, x) where f: fn(f64) -> f64, x: f64 -> f64
            "grad" => HirType::Fn {
                params: vec![
                    HirType::Fn {
                        params: vec![HirType::Dual],
                        return_type: Box::new(HirType::Dual),
                    },
                    HirType::F64,
                ],
                return_type: Box::new(HirType::F64),
            },
            // Compute Jacobian of vector function (returns matrix of partial derivatives)
            // jacobian(f, x) where f: fn(vec) -> vec, x: vec -> mat
            "jacobian" => HirType::Fn {
                params: vec![
                    HirType::Fn {
                        params: vec![HirType::Array {
                            element: Box::new(HirType::Dual),
                            size: None,
                        }],
                        return_type: Box::new(HirType::Array {
                            element: Box::new(HirType::Dual),
                            size: None,
                        }),
                    },
                    HirType::Array {
                        element: Box::new(HirType::F64),
                        size: None,
                    },
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Array {
                        element: Box::new(HirType::F64),
                        size: None,
                    }),
                    size: None,
                }),
            },
            // Compute Hessian (second derivatives) of scalar function
            // hessian(f, x) where f: fn(vec) -> scalar, x: vec -> mat
            "hessian" => HirType::Fn {
                params: vec![
                    HirType::Fn {
                        params: vec![HirType::Array {
                            element: Box::new(HirType::Dual),
                            size: None,
                        }],
                        return_type: Box::new(HirType::Dual),
                    },
                    HirType::Array {
                        element: Box::new(HirType::F64),
                        size: None,
                    },
                ],
                return_type: Box::new(HirType::Array {
                    element: Box::new(HirType::Array {
                        element: Box::new(HirType::F64),
                        size: None,
                    }),
                    size: None,
                }),
            },

            // ==================== FFI / RAW POINTER OPERATIONS ====================
            // Create null const pointer
            "null_ptr" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Create null mut pointer
            "null_mut" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Check if pointer is null
            "is_null" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::Bool),
            },
            // Compare two pointers
            "ptr_eq" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                ],
                return_type: Box::new(HirType::Bool),
            },
            // Get address as integer
            "ptr_addr" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::I64),
            },
            // Create const pointer from address
            "ptr_from_addr" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Create mut pointer from address
            "ptr_from_addr_mut" => HirType::Fn {
                params: vec![HirType::I64],
                return_type: Box::new(HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Offset pointer by bytes
            "ptr_offset" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::I64,
                ],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Add elements to pointer
            "ptr_add" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::I64,
                ],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Subtract elements from pointer
            "ptr_sub" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::I64,
                ],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Difference between pointers
            "ptr_diff" => HirType::Fn {
                params: vec![
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                    HirType::RawPointer {
                        mutable: false,
                        inner: Box::new(HirType::Unit),
                    },
                ],
                return_type: Box::new(HirType::I64),
            },
            // Cast *mut to *const
            "as_const" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Cast *const to *mut (unsafe)
            "as_mut" => HirType::Fn {
                params: vec![HirType::RawPointer {
                    mutable: false,
                    inner: Box::new(HirType::Unit),
                }],
                return_type: Box::new(HirType::RawPointer {
                    mutable: true,
                    inner: Box::new(HirType::Unit),
                }),
            },
            // Get size of type
            "size_of" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::I64),
            },
            // Get alignment of type
            "align_of" => HirType::Fn {
                params: vec![],
                return_type: Box::new(HirType::I64),
            },

            _ => HirType::Error,
        }
    }

    /// Check if a name is a builtin enum variant
    fn is_builtin_variant(&self, name: &str) -> bool {
        matches!(name, "None" | "Some" | "Ok" | "Err")
    }

    /// Get the type of a builtin variant, using expected type for inference
    fn get_builtin_variant_type(&self, name: &str, expected: Option<&Type>) -> HirType {
        match name {
            "None" => {
                // If we have an expected type that's Option<T>, use that
                if let Some(Type::Named {
                    name: type_name,
                    args,
                }) = expected
                {
                    if type_name == "Option" {
                        return HirType::Named {
                            name: "Option".to_string(),
                            args: args.iter().map(|t| self.type_to_hir(t)).collect(),
                        };
                    }
                }
                // Default to Option<()>
                HirType::Named {
                    name: "Option".to_string(),
                    args: vec![HirType::Unit],
                }
            }
            "Some" => {
                // Some is a constructor function - for now return a generic Option type
                HirType::Fn {
                    params: vec![HirType::Unit], // Takes one arg
                    return_type: Box::new(HirType::Named {
                        name: "Option".to_string(),
                        args: vec![HirType::Unit],
                    }),
                }
            }
            "Ok" => {
                // Ok is a constructor for Result<T, E>
                HirType::Fn {
                    params: vec![HirType::Unit],
                    return_type: Box::new(HirType::Named {
                        name: "Result".to_string(),
                        args: vec![HirType::Unit, HirType::Unit],
                    }),
                }
            }
            "Err" => {
                // Err is a constructor for Result<T, E>
                HirType::Fn {
                    params: vec![HirType::Unit],
                    return_type: Box::new(HirType::Named {
                        name: "Result".to_string(),
                        args: vec![HirType::Unit, HirType::Unit],
                    }),
                }
            }
            _ => HirType::Error,
        }
    }

    /// Parse a unit string (e.g., "mg", "mL/min") into HirUnit
    fn parse_unit_string(&self, unit_str: &str) -> HirUnit {
        // Handle compound units with / and *
        if let Some(pos) = unit_str.find('/') {
            let num = &unit_str[..pos];
            let den = &unit_str[pos + 1..];
            let num_unit = self.parse_unit_string(num);
            let den_unit = self.parse_unit_string(den);
            return num_unit.divide(&den_unit);
        }
        if let Some(pos) = unit_str.find('*') {
            let left = &unit_str[..pos];
            let right = &unit_str[pos + 1..];
            let left_unit = self.parse_unit_string(left);
            let right_unit = self.parse_unit_string(right);
            return left_unit.multiply(&right_unit);
        }
        // Simple unit
        HirUnit::simple(unit_str)
    }

    fn binary_result_type(&self, op: BinaryOp, left: &HirType, right: &HirType) -> HirType {
        match op {
            BinaryOp::Add
            | BinaryOp::Sub
            | BinaryOp::Mul
            | BinaryOp::Div
            | BinaryOp::Rem
            | BinaryOp::PlusMinus => left.clone(),
            BinaryOp::Eq
            | BinaryOp::Ne
            | BinaryOp::Lt
            | BinaryOp::Le
            | BinaryOp::Gt
            | BinaryOp::Ge
            | BinaryOp::And
            | BinaryOp::Or => HirType::Bool,
            BinaryOp::BitAnd
            | BinaryOp::BitOr
            | BinaryOp::BitXor
            | BinaryOp::Shl
            | BinaryOp::Shr => left.clone(),
            // Concatenation: combine array sizes
            BinaryOp::Concat => match (left, right) {
                (
                    HirType::Array {
                        element: left_elem,
                        size: left_size,
                    },
                    HirType::Array {
                        element: _right_elem,
                        size: right_size,
                    },
                ) => {
                    let combined_size = match (left_size, right_size) {
                        (Some(l), Some(r)) => Some(l + r),
                        _ => None,
                    };
                    HirType::Array {
                        element: left_elem.clone(),
                        size: combined_size,
                    }
                }
                _ => left.clone(),
            },
        }
    }

    fn unary_result_type(&self, op: UnaryOp, operand: &HirType) -> HirType {
        match op {
            UnaryOp::Neg => operand.clone(),
            UnaryOp::Not => {
                if *operand == HirType::Bool {
                    HirType::Bool
                } else {
                    operand.clone()
                }
            }
            UnaryOp::Ref => HirType::Ref {
                mutable: false,
                inner: Box::new(operand.clone()),
            },
            UnaryOp::RefMut => HirType::Ref {
                mutable: true,
                inner: Box::new(operand.clone()),
            },
            UnaryOp::Deref => {
                if let HirType::Ref { inner, .. } = operand {
                    *inner.clone()
                } else {
                    HirType::Error
                }
            }
        }
    }

    fn lower_binary_op(&self, op: BinaryOp) -> HirBinaryOp {
        match op {
            BinaryOp::Add => HirBinaryOp::Add,
            BinaryOp::Sub => HirBinaryOp::Sub,
            BinaryOp::Mul => HirBinaryOp::Mul,
            BinaryOp::Div => HirBinaryOp::Div,
            BinaryOp::Rem => HirBinaryOp::Rem,
            BinaryOp::Eq => HirBinaryOp::Eq,
            BinaryOp::Ne => HirBinaryOp::Ne,
            BinaryOp::Lt => HirBinaryOp::Lt,
            BinaryOp::Le => HirBinaryOp::Le,
            BinaryOp::Gt => HirBinaryOp::Gt,
            BinaryOp::Ge => HirBinaryOp::Ge,
            BinaryOp::And => HirBinaryOp::And,
            BinaryOp::Or => HirBinaryOp::Or,
            BinaryOp::BitAnd => HirBinaryOp::BitAnd,
            BinaryOp::BitOr => HirBinaryOp::BitOr,
            BinaryOp::BitXor => HirBinaryOp::BitXor,
            BinaryOp::Shl => HirBinaryOp::Shl,
            BinaryOp::Shr => HirBinaryOp::Shr,
            BinaryOp::PlusMinus => HirBinaryOp::PlusMinus,
            BinaryOp::Concat => HirBinaryOp::Concat,
        }
    }

    fn lower_unary_op(&self, op: UnaryOp) -> HirUnaryOp {
        match op {
            UnaryOp::Neg => HirUnaryOp::Neg,
            UnaryOp::Not => HirUnaryOp::Not,
            UnaryOp::Ref => HirUnaryOp::Ref,
            UnaryOp::RefMut => HirUnaryOp::RefMut,
            UnaryOp::Deref => HirUnaryOp::Deref,
        }
    }

    /// Evaluate a constant expression to a usize (for array sizes)
    fn eval_const_usize(&self, expr: &Expr) -> Option<usize> {
        match expr {
            Expr::Literal {
                value: Literal::Int(i),
                ..
            } if *i >= 0 => Some(*i as usize),
            _ => None, // Non-literal const expressions not yet supported
        }
    }

    fn lower_type_expr(&mut self, ty: &TypeExpr) -> Type {
        match ty {
            TypeExpr::Unit => Type::Unit,
            TypeExpr::Named { path, args, unit } => {
                let base_type = if path.segments.len() == 1 {
                    let name = &path.segments[0];
                    match name.as_str() {
                        "bool" => Type::Bool,
                        "i8" => Type::I8,
                        "i16" => Type::I16,
                        "i32" => Type::I32,
                        "i64" => Type::I64,
                        "i128" => Type::I128,
                        "isize" => Type::Isize,
                        "u8" => Type::U8,
                        "u16" => Type::U16,
                        "u32" => Type::U32,
                        "u64" => Type::U64,
                        "u128" => Type::U128,
                        "usize" => Type::Usize,
                        "f32" => Type::F32,
                        "f64" => Type::F64,
                        "char" => Type::Char,
                        "str" => Type::Str,
                        "String" => Type::String,
                        // Linear algebra primitives
                        "vec2" => Type::Vec2,
                        "vec3" => Type::Vec3,
                        "vec4" => Type::Vec4,
                        "mat2" => Type::Mat2,
                        "mat3" => Type::Mat3,
                        "mat4" => Type::Mat4,
                        "quat" => Type::Quat,
                        "dual" => Type::Dual,
                        _ => Type::Named {
                            name: name.clone(),
                            args: args.iter().map(|a| self.lower_type_expr(a)).collect(),
                        },
                    }
                } else {
                    Type::Named {
                        name: path.to_string(),
                        args: args.iter().map(|a| self.lower_type_expr(a)).collect(),
                    }
                };
                // If there's a unit annotation, wrap in Quantity type
                if let Some(unit_str) = unit {
                    Type::Quantity {
                        numeric: Box::new(base_type),
                        unit: unit_str.clone(),
                    }
                } else {
                    base_type
                }
            }
            TypeExpr::Reference { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.lower_type_expr(inner)),
            },
            TypeExpr::RawPointer { mutable, inner } => Type::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.lower_type_expr(inner)),
            },
            TypeExpr::Array { element, size } => Type::Array {
                element: Box::new(self.lower_type_expr(element)),
                size: size.as_ref().and_then(|s| self.eval_const_usize(s)),
            },
            TypeExpr::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.lower_type_expr(e)).collect())
            }
            TypeExpr::Function {
                params,
                return_type,
                ..
            } => Type::Function {
                params: params.iter().map(|p| self.lower_type_expr(p)).collect(),
                return_type: Box::new(self.lower_type_expr(return_type)),
                effects: types::EffectSet::new(),
            },
            TypeExpr::Infer => Type::Unknown,
            TypeExpr::SelfType => Type::SelfType,

            // Epistemic types - map to Unknown for now, will be properly implemented later
            TypeExpr::Knowledge { value_type, .. } => {
                // For now, treat Knowledge[T] as just T for type checking purposes
                self.lower_type_expr(value_type)
            }
            TypeExpr::Quantity { numeric_type, .. } => {
                // For now, treat Quantity[T, unit] as just T
                self.lower_type_expr(numeric_type)
            }
            TypeExpr::Tensor { element_type, .. } => {
                // Tensor becomes array-like
                Type::Array {
                    element: Box::new(self.lower_type_expr(element_type)),
                    size: None,
                }
            }
            TypeExpr::Ontology { ontology, term } => {
                // Track that this ontology prefix is used
                self.used_ontology_prefixes.insert(ontology.clone());
                // Ontology term as a semantic type
                Type::Ontology {
                    namespace: ontology.clone(),
                    term: term.clone().unwrap_or_default(),
                }
            }
            TypeExpr::Linear { inner, .. } => {
                // Linear types pass through the inner type
                self.lower_type_expr(inner)
            }
            TypeExpr::Effected { inner, .. } => {
                // Effected types pass through the inner type
                self.lower_type_expr(inner)
            }
            TypeExpr::Tile { element_type, .. } => {
                // Tile becomes array-like (for type checking purposes)
                Type::Array {
                    element: Box::new(self.lower_type_expr(element_type)),
                    size: None,
                }
            }
        }
    }

    fn type_to_hir(&self, ty: &Type) -> HirType {
        match ty {
            Type::Unit => HirType::Unit,
            Type::Bool => HirType::Bool,
            Type::I8 => HirType::I8,
            Type::I16 => HirType::I16,
            Type::I32 => HirType::I32,
            Type::I64 => HirType::I64,
            Type::I128 => HirType::I128,
            Type::Isize => HirType::Isize,
            Type::U8 => HirType::U8,
            Type::U16 => HirType::U16,
            Type::U32 => HirType::U32,
            Type::U64 => HirType::U64,
            Type::U128 => HirType::U128,
            Type::Usize => HirType::Usize,
            Type::F32 => HirType::F32,
            Type::F64 => HirType::F64,
            Type::Char => HirType::Char,
            Type::Str | Type::String => HirType::String,
            Type::Ref { mutable, inner, .. } => HirType::Ref {
                mutable: *mutable,
                inner: Box::new(self.type_to_hir(inner)),
            },
            Type::RawPointer { mutable, inner } => HirType::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.type_to_hir(inner)),
            },
            Type::Array { element, size } => HirType::Array {
                element: Box::new(self.type_to_hir(element)),
                size: *size,
            },
            Type::Tuple(elems) => {
                HirType::Tuple(elems.iter().map(|e| self.type_to_hir(e)).collect())
            }
            Type::Function {
                params,
                return_type,
                ..
            } => HirType::Fn {
                params: params.iter().map(|p| self.type_to_hir(p)).collect(),
                return_type: Box::new(self.type_to_hir(return_type)),
            },
            Type::Named { name, args } => HirType::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.type_to_hir(a)).collect(),
            },
            Type::Quantity { numeric, unit } => HirType::Quantity {
                numeric: Box::new(self.type_to_hir(numeric)),
                unit: self.parse_unit_string(unit),
            },
            Type::Var(v) => HirType::Var(v.0),
            Type::Forall { inner, .. } => self.type_to_hir(inner),
            Type::Ontology { namespace, term } => HirType::Ontology {
                namespace: namespace.clone(),
                term: term.clone(),
            },
            Type::Never | Type::Unknown | Type::Error | Type::SelfType => HirType::Error,
            // Linear algebra primitives
            Type::Vec2 => HirType::Vec2,
            Type::Vec3 => HirType::Vec3,
            Type::Vec4 => HirType::Vec4,
            Type::Mat2 => HirType::Mat2,
            Type::Mat3 => HirType::Mat3,
            Type::Mat4 => HirType::Mat4,
            Type::Quat => HirType::Quat,
            // Automatic differentiation
            Type::Dual => HirType::Dual,
        }
    }

    fn hir_type_to_type(&self, ty: &HirType) -> Type {
        match ty {
            HirType::Unit => Type::Unit,
            HirType::Bool => Type::Bool,
            HirType::I8 => Type::I8,
            HirType::I16 => Type::I16,
            HirType::I32 => Type::I32,
            HirType::I64 => Type::I64,
            HirType::I128 => Type::I128,
            HirType::Isize => Type::Isize,
            HirType::U8 => Type::U8,
            HirType::U16 => Type::U16,
            HirType::U32 => Type::U32,
            HirType::U64 => Type::U64,
            HirType::U128 => Type::U128,
            HirType::Usize => Type::Usize,
            HirType::F32 => Type::F32,
            HirType::F64 => Type::F64,
            HirType::Char => Type::Char,
            HirType::String => Type::String,
            HirType::Ref { mutable, inner } => Type::Ref {
                mutable: *mutable,
                lifetime: None,
                inner: Box::new(self.hir_type_to_type(inner)),
            },
            HirType::RawPointer { mutable, inner } => Type::RawPointer {
                mutable: *mutable,
                inner: Box::new(self.hir_type_to_type(inner)),
            },
            HirType::Array { element, size } => Type::Array {
                element: Box::new(self.hir_type_to_type(element)),
                size: *size,
            },
            HirType::Tuple(elems) => {
                Type::Tuple(elems.iter().map(|e| self.hir_type_to_type(e)).collect())
            }
            HirType::Named { name, args } => Type::Named {
                name: name.clone(),
                args: args.iter().map(|a| self.hir_type_to_type(a)).collect(),
            },
            HirType::Fn {
                params,
                return_type,
            } => Type::Function {
                params: params.iter().map(|p| self.hir_type_to_type(p)).collect(),
                return_type: Box::new(self.hir_type_to_type(return_type)),
                effects: types::EffectSet::new(),
            },
            HirType::Var(v) => Type::Var(TypeVar(*v)),
            HirType::Never => Type::Never,
            HirType::Error => Type::Error,

            // Epistemic types - map back to their inner types for now
            HirType::Knowledge { inner, .. } => self.hir_type_to_type(inner),
            HirType::Quantity { numeric, unit } => Type::Quantity {
                numeric: Box::new(self.hir_type_to_type(numeric)),
                unit: unit.format(),
            },
            HirType::Tensor { element, .. } => Type::Array {
                element: Box::new(self.hir_type_to_type(element)),
                size: None,
            },
            HirType::Ontology { namespace, term } => Type::Ontology {
                namespace: namespace.clone(),
                term: term.clone(),
            },
            // Linear algebra primitives
            HirType::Vec2 => Type::Vec2,
            HirType::Vec3 => Type::Vec3,
            HirType::Vec4 => Type::Vec4,
            HirType::Mat2 => Type::Mat2,
            HirType::Mat3 => Type::Mat3,
            HirType::Mat4 => Type::Mat4,
            HirType::Quat => Type::Quat,
            // Automatic differentiation
            HirType::Dual => Type::Dual,
        }
    }

    fn pattern_name(&self, pattern: &Pattern) -> String {
        match pattern {
            Pattern::Binding { name, .. } => name.clone(),
            Pattern::Wildcard => "_".to_string(),
            _ => "_".to_string(),
        }
    }

    /// Lower AST provenance marker to HIR provenance
    fn lower_provenance(&self, prov: &ProvenanceMarker) -> HirProvenance {
        match prov.kind {
            ProvenanceKind::Derived => HirProvenance::Derived { sources: vec![] },
            ProvenanceKind::Source => HirProvenance::Measured {
                source: "source".to_string(),
            },
            ProvenanceKind::Computed => HirProvenance::Derived {
                sources: vec!["computed".to_string()],
            },
            ProvenanceKind::Literature => HirProvenance::PeerReviewed {
                citation: String::new(),
            },
            ProvenanceKind::Measured => HirProvenance::Measured {
                source: "measurement".to_string(),
            },
            ProvenanceKind::Input => HirProvenance::UserInput,
        }
    }

    fn solve_constraints(&mut self) -> Result<()> {
        // Simple unification - a real implementation would be more sophisticated
        // Collect errors first to avoid borrow issues
        let errors: Vec<_> = self
            .constraints
            .iter()
            .filter(|c| !self.types_compatible(&c.expected, &c.actual))
            .map(|c| {
                (
                    format!(
                        "Type mismatch: expected {:?}, found {:?}",
                        c.expected, c.actual
                    ),
                    c.span,
                )
            })
            .collect();

        for (msg, span) in errors {
            self.errors.push(TypeError {
                message: msg,
                span,
                code: "E0308".to_string(),
            });
        }
        Ok(())
    }

    /// Get the return type of a method call based on receiver type and method name
    fn get_method_return_type(
        &self,
        receiver_ty: &HirType,
        method: &str,
        _args: &[HirExpr],
    ) -> HirType {
        match receiver_ty {
            // Vec<T> methods
            HirType::Named { name, args } if name == "Vec" => {
                match method {
                    "is_empty" => HirType::Bool,
                    "len" => HirType::Usize,
                    "first" | "last" => {
                        // Returns Option<&T>
                        if let Some(elem_ty) = args.first() {
                            HirType::Named {
                                name: "Option".to_string(),
                                args: vec![elem_ty.clone()],
                            }
                        } else {
                            HirType::Error
                        }
                    }
                    "get" => {
                        // Returns Option<&T>
                        if let Some(elem_ty) = args.first() {
                            HirType::Named {
                                name: "Option".to_string(),
                                args: vec![elem_ty.clone()],
                            }
                        } else {
                            HirType::Error
                        }
                    }
                    "push" | "pop" | "clear" | "remove" | "insert" => HirType::Unit,
                    "contains" => HirType::Bool,
                    "iter" => receiver_ty.clone(), // Simplified - would be Iterator<T>
                    _ => HirType::Error,
                }
            }
            // String methods
            HirType::String => match method {
                "len" => HirType::Usize,
                "is_empty" => HirType::Bool,
                "contains" | "starts_with" | "ends_with" => HirType::Bool,
                "trim" | "to_lowercase" | "to_uppercase" => HirType::String,
                "chars" | "bytes" => HirType::Error, // Would be iterator
                _ => HirType::Error,
            },
            // Option<T> methods
            HirType::Named { name, args } if name == "Option" => match method {
                "is_some" | "is_none" => HirType::Bool,
                "unwrap" | "expect" => {
                    if let Some(inner) = args.first() {
                        inner.clone()
                    } else {
                        HirType::Error
                    }
                }
                "unwrap_or" | "unwrap_or_else" => {
                    if let Some(inner) = args.first() {
                        inner.clone()
                    } else {
                        HirType::Error
                    }
                }
                _ => HirType::Error,
            },
            // Result<T, E> methods
            HirType::Named { name, args } if name == "Result" => match method {
                "is_ok" | "is_err" => HirType::Bool,
                "unwrap" | "expect" => {
                    if let Some(ok_ty) = args.first() {
                        ok_ty.clone()
                    } else {
                        HirType::Error
                    }
                }
                "unwrap_err" | "expect_err" => {
                    if args.len() > 1 {
                        args[1].clone()
                    } else {
                        HirType::Error
                    }
                }
                _ => HirType::Error,
            },
            // Default - unknown method
            _ => HirType::Error,
        }
    }

    /// Lower an AST Pattern to an HIR Pattern
    fn lower_pattern(&self, pattern: &Pattern) -> HirPattern {
        match pattern {
            Pattern::Wildcard => HirPattern::Wildcard,
            Pattern::Literal(lit) => {
                let (hir_lit, _) = self.check_literal_with_expected(lit, None);
                HirPattern::Literal(hir_lit)
            }
            Pattern::Binding { name, mutable } => HirPattern::Binding {
                name: name.clone(),
                mutable: *mutable,
            },
            Pattern::Tuple(patterns) => {
                HirPattern::Tuple(patterns.iter().map(|p| self.lower_pattern(p)).collect())
            }
            Pattern::Struct { path, fields } => HirPattern::Struct {
                name: path.segments.last().cloned().unwrap_or_default(),
                fields: fields
                    .iter()
                    .map(|(name, pat)| (name.clone(), self.lower_pattern(pat)))
                    .collect(),
            },
            Pattern::Enum { path, patterns } => {
                let segments = &path.segments;
                let (enum_name, variant) = if segments.len() >= 2 {
                    (
                        segments[segments.len() - 2].clone(),
                        segments[segments.len() - 1].clone(),
                    )
                } else {
                    (String::new(), segments.last().cloned().unwrap_or_default())
                };
                HirPattern::Variant {
                    enum_name,
                    variant,
                    patterns: patterns
                        .as_ref()
                        .map(|ps| ps.iter().map(|p| self.lower_pattern(p)).collect())
                        .unwrap_or_default(),
                }
            }
            Pattern::Or(patterns) => {
                HirPattern::Or(patterns.iter().map(|p| self.lower_pattern(p)).collect())
            }
        }
    }

    fn types_compatible(&self, t1: &Type, t2: &Type) -> bool {
        match (t1, t2) {
            (Type::Var(_), _) | (_, Type::Var(_)) => true, // Type variables unify with anything
            (Type::Unknown, _) | (_, Type::Unknown) => true,
            (Type::Error, _) | (_, Type::Error) => true,
            (Type::Never, _) | (_, Type::Never) => true, // Never is subtype of all types
            (Type::Unit, Type::Unit) => true,
            (Type::Bool, Type::Bool) => true,
            (Type::I8, Type::I8) => true,
            (Type::I16, Type::I16) => true,
            (Type::I32, Type::I32) => true,
            (Type::I64, Type::I64) => true,
            (Type::I128, Type::I128) => true,
            (Type::Isize, Type::Isize) => true,
            (Type::U8, Type::U8) => true,
            (Type::U16, Type::U16) => true,
            (Type::U32, Type::U32) => true,
            (Type::U64, Type::U64) => true,
            (Type::U128, Type::U128) => true,
            (Type::Usize, Type::Usize) => true,
            (Type::F32, Type::F32) => true,
            (Type::F64, Type::F64) => true,
            (Type::Char, Type::Char) => true,
            (Type::Str, Type::Str) => true,
            (Type::String, Type::String) => true,
            (
                Type::Ref {
                    mutable: m1,
                    inner: i1,
                    ..
                },
                Type::Ref {
                    mutable: m2,
                    inner: i2,
                    ..
                },
            ) => m1 == m2 && self.types_compatible(i1, i2),
            (
                Type::Array {
                    element: e1,
                    size: s1,
                },
                Type::Array {
                    element: e2,
                    size: s2,
                },
            ) => {
                // Size compatibility: None (unknown) matches any size
                let size_ok = match (s1, s2) {
                    (None, _) | (_, None) => true, // Unknown size is compatible with any size
                    (Some(a), Some(b)) => a == b,  // Known sizes must match
                };
                size_ok && self.types_compatible(e1, e2)
            }
            (Type::Tuple(t1), Type::Tuple(t2)) => {
                t1.len() == t2.len()
                    && t1
                        .iter()
                        .zip(t2.iter())
                        .all(|(a, b)| self.types_compatible(a, b))
            }
            (Type::Named { name: n1, args: a1 }, Type::Named { name: n2, args: a2 }) => {
                n1 == n2
                    && a1.len() == a2.len()
                    && a1
                        .iter()
                        .zip(a2.iter())
                        .all(|(a, b)| self.types_compatible(a, b))
            }
            // Quantity type compatibility - same unit and compatible numeric types
            (
                Type::Quantity {
                    numeric: n1,
                    unit: u1,
                },
                Type::Quantity {
                    numeric: n2,
                    unit: u2,
                },
            ) => u1 == u2 && self.types_compatible(n1, n2),
            // Quantity with plain numeric - allow if numeric types match (implicit unit stripping)
            (Type::Quantity { numeric, .. }, other) | (other, Type::Quantity { numeric, .. }) => {
                self.types_compatible(numeric, other)
            }
            // Ontology type compatibility - check if within default threshold
            (
                Type::Ontology {
                    namespace: ns1,
                    term: t1,
                },
                Type::Ontology {
                    namespace: ns2,
                    term: t2,
                },
            ) => {
                // Same type = compatible
                if ns1 == ns2 && t1 == t2 {
                    return true;
                }
                // Same namespace = compatible (within same ontology)
                if ns1 == ns2 {
                    return true;
                }
                // Check alignment
                let key1 = format!("{}:{}", ns1, t1);
                let key2 = format!("{}:{}", ns2, t2);
                self.get_semantic_distance(&key1, &key2)
                    .map(|d| d <= self.default_threshold)
                    .unwrap_or(false)
            }
            // Named type (alias) compared with Ontology type - resolve alias
            (Type::Named { name, .. }, Type::Ontology { namespace, term }) => {
                if let Some(TypeDef::Alias(alias_ty, _, _)) = self.type_defs.get(name) {
                    if let Type::Ontology {
                        namespace: alias_ns,
                        term: alias_term,
                    } = alias_ty
                    {
                        // Same namespace = compatible
                        if alias_ns == namespace {
                            return true;
                        }
                        // Check alignment
                        let key1 = format!("{}:{}", alias_ns, alias_term);
                        let key2 = format!("{}:{}", namespace, term);
                        return self
                            .get_semantic_distance(&key1, &key2)
                            .map(|d| d <= self.default_threshold)
                            .unwrap_or(false);
                    }
                }
                false
            }
            // Ontology type compared with Named type (alias) - resolve alias
            (Type::Ontology { namespace, term }, Type::Named { name, .. }) => {
                if let Some(TypeDef::Alias(alias_ty, _, _)) = self.type_defs.get(name) {
                    if let Type::Ontology {
                        namespace: alias_ns,
                        term: alias_term,
                    } = alias_ty
                    {
                        // Same namespace = compatible
                        if alias_ns == namespace {
                            return true;
                        }
                        // Check alignment
                        let key1 = format!("{}:{}", namespace, term);
                        let key2 = format!("{}:{}", alias_ns, alias_term);
                        return self
                            .get_semantic_distance(&key1, &key2)
                            .map(|d| d <= self.default_threshold)
                            .unwrap_or(false);
                    }
                }
                false
            }
            _ => false,
        }
    }
}

impl TypeEnv {
    fn push_scope(&mut self) {
        self.scopes.push(Scope::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: String, ty: Type, mutable: bool) {
        self.bind_with_module(name, ty, mutable, None);
    }

    /// Bind a name with explicit module origin
    fn bind_with_module(
        &mut self,
        name: String,
        ty: Type,
        mutable: bool,
        source_module: Option<ModuleId>,
    ) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.bindings.insert(
                name,
                TypeBinding {
                    ty,
                    mutable,
                    used: false,
                    source_module,
                },
            );
        }
    }

    /// Bind a module-qualified name (e.g., math.sin)
    fn bind_qualified(&mut self, module_path: Vec<String>, name: String, ty: Type, mutable: bool) {
        let module_id = ModuleId::new(module_path.clone());
        self.module_bindings.insert(
            (module_path, name),
            TypeBinding {
                ty,
                mutable,
                used: false,
                source_module: Some(module_id),
            },
        );
    }

    fn lookup(&self, name: &str) -> Option<&TypeBinding> {
        for scope in self.scopes.iter().rev() {
            if let Some(binding) = scope.bindings.get(name) {
                return Some(binding);
            }
        }
        None
    }

    /// Lookup a qualified path (e.g., ["math", "sin"])
    fn lookup_qualified(&self, path: &[String]) -> Option<&TypeBinding> {
        if path.len() <= 1 {
            return self.lookup(path.first().map(|s| s.as_str()).unwrap_or(""));
        }

        // Split into module path and name
        let module_path = &path[..path.len() - 1];
        let name = &path[path.len() - 1];

        // Try exact module path match
        if let Some(binding) = self
            .module_bindings
            .get(&(module_path.to_vec(), name.clone()))
        {
            return Some(binding);
        }

        // Try looking up the full qualified name (e.g., "String::new" for associated functions)
        let full_name = path.join("::");
        if let Some(binding) = self.lookup(&full_name) {
            return Some(binding);
        }

        // Fall back to unqualified lookup (for local names)
        self.lookup(name)
    }

    fn lookup_mut(&mut self, name: &str) -> Option<&mut TypeBinding> {
        for scope in self.scopes.iter_mut().rev() {
            if let Some(binding) = scope.bindings.get_mut(name) {
                return Some(binding);
            }
        }
        None
    }
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}
