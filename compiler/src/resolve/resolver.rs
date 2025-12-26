//! Name resolution pass

use super::symbols::*;
use crate::ast::{ModuleId, *};
use crate::common::{NodeId, Span};
use miette::{Diagnostic, Result, SourceSpan};
use thiserror::Error;

/// Resolution error
#[derive(Error, Debug, Diagnostic)]
pub enum ResolveError {
    #[error("Undefined variable: {name}")]
    UndefinedVar {
        name: String,
        #[label("not found in scope")]
        span: SourceSpan,
    },

    #[error("Undefined type: {name}{}", suggestion.as_ref().map(|s| format!(". Did you mean `{}`?", s)).unwrap_or_default())]
    UndefinedType {
        name: String,
        suggestion: Option<String>,
        #[label("type not found")]
        span: SourceSpan,
    },

    #[error("Duplicate definition: {name}")]
    DuplicateDef {
        name: String,
        #[label("already defined")]
        span: SourceSpan,
    },

    #[error("Cannot use {name} as a value")]
    NotAValue {
        name: String,
        #[label("this is a type, not a value")]
        span: SourceSpan,
    },

    #[error("Cannot use {name} as a type")]
    NotAType {
        name: String,
        #[label("this is a value, not a type")]
        span: SourceSpan,
    },
}

/// Resolved AST (AST + symbol table)
#[derive(Debug)]
pub struct ResolvedAst {
    pub ast: Ast,
    pub symbols: SymbolTable,
}

/// Resolve names in an AST
pub fn resolve(ast: Ast) -> Result<ResolvedAst> {
    let resolver = Resolver::new();
    resolver.resolve(ast)
}

/// Name resolver
pub struct Resolver {
    symbols: SymbolTable,
    errors: Vec<ResolveError>,
}

impl Resolver {
    pub fn new() -> Self {
        Self {
            symbols: SymbolTable::new(),
            errors: Vec::new(),
        }
    }

    /// Resolve all names in the AST
    pub fn resolve(mut self, ast: Ast) -> Result<ResolvedAst> {
        // First pass: collect all top-level definitions
        for item in &ast.items {
            self.collect_item(item);
        }

        // Second pass: resolve bodies
        for item in &ast.items {
            self.resolve_item(item);
        }

        if !self.errors.is_empty() {
            let messages: Vec<_> = self.errors.iter().map(|e| e.to_string()).collect();
            return Err(miette::miette!(
                "Resolution errors:\n{}",
                messages.join("\n")
            ));
        }

        Ok(ResolvedAst {
            ast,
            symbols: self.symbols,
        })
    }

    /// First pass: collect definitions
    fn collect_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.define_function(f),
            Item::Struct(s) => self.define_struct(s),
            Item::Enum(e) => self.define_enum(e),
            Item::TypeAlias(t) => self.define_type_alias(t),
            Item::Effect(e) => self.define_effect(e),
            Item::Trait(t) => self.define_trait(t),
            Item::Global(g) => self.define_global(g),
            Item::Module(m) => self.collect_module(m),
            Item::Import(i) => self.collect_import(i),
            _ => {}
        }
    }

    fn collect_module(&mut self, m: &ModuleDef) {
        // Create a module ID from the name
        let parent_path = self.symbols.current_module().path.clone();
        let mut module_path = parent_path;
        module_path.push(m.name.clone());
        let module_id = ModuleId::new(module_path);

        // Enter the module
        self.symbols.enter_module(module_id.clone());

        // Define the module as a symbol
        let def_id = self.symbols.fresh_def_id();
        self.symbols.insert(Symbol {
            def_id,
            name: m.name.clone(),
            kind: DefKind::Module,
            node_id: m.id,
            span: m.span,
            parent: None,
        });

        // Recursively collect items if inline module
        if let Some(ref items) = m.items {
            for item in items {
                self.collect_item(item);
            }
        }

        // Exit module
        self.symbols.exit_module();
    }

    fn collect_import(&mut self, i: &ImportDef) {
        // Process import into the current module's symbol table
        let path: Vec<String> = i.path.segments.clone();
        if let Err(e) = self
            .symbols
            .process_import(&path, i.items.as_deref(), i.is_reexport)
        {
            // Imports to unknown modules are common during initial parsing
            // Don't treat as error yet - module may be loaded later
            // TODO: Track unresolved imports and verify after all modules loaded
            let _ = e; // Silence warning for now
        }
    }

    fn define_function(&mut self, f: &FnDef) {
        let def_id = self.symbols.fresh_def_id();

        if let Err(_) = self.symbols.define(f.name.clone(), def_id) {
            self.errors.push(ResolveError::DuplicateDef {
                name: f.name.clone(),
                span: self.span_to_source(f.span),
            });
            return;
        }

        self.symbols.insert(Symbol {
            def_id,
            name: f.name.clone(),
            kind: DefKind::Function,
            node_id: f.id,
            span: f.span,
            parent: None,
        });
    }

    fn define_struct(&mut self, s: &StructDef) {
        let def_id = self.symbols.fresh_def_id();

        if let Err(_) = self.symbols.define_type(s.name.clone(), def_id) {
            self.errors.push(ResolveError::DuplicateDef {
                name: s.name.clone(),
                span: self.span_to_source(s.span),
            });
            return;
        }

        self.symbols.insert(Symbol {
            def_id,
            name: s.name.clone(),
            kind: DefKind::Struct {
                is_linear: s.modifiers.linear,
                is_affine: s.modifiers.affine,
            },
            node_id: s.id,
            span: s.span,
            parent: None,
        });
    }

    fn define_enum(&mut self, e: &EnumDef) {
        let def_id = self.symbols.fresh_def_id();

        let _ = self.symbols.define_type(e.name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name: e.name.clone(),
            kind: DefKind::Enum {
                is_linear: e.modifiers.linear,
                is_affine: e.modifiers.affine,
            },
            node_id: e.id,
            span: e.span,
            parent: None,
        });

        // Define variants as values in the value namespace
        for variant in &e.variants {
            let variant_def_id = self.symbols.fresh_def_id();
            let variant_name = format!("{}::{}", e.name, variant.name);
            let _ = self.symbols.define(variant_name.clone(), variant_def_id);
            self.symbols.insert(Symbol {
                def_id: variant_def_id,
                name: variant_name,
                kind: DefKind::Variant,
                node_id: variant.id,
                span: e.span,
                parent: Some(def_id),
            });
        }
    }

    fn define_type_alias(&mut self, t: &TypeAliasDef) {
        let def_id = self.symbols.fresh_def_id();

        let _ = self.symbols.define_type(t.name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name: t.name.clone(),
            kind: DefKind::TypeAlias,
            node_id: t.id,
            span: t.span,
            parent: None,
        });
    }

    fn define_effect(&mut self, e: &EffectDef) {
        let def_id = self.symbols.fresh_def_id();

        // Effects go in type namespace
        let _ = self.symbols.define_type(e.name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name: e.name.clone(),
            kind: DefKind::Effect,
            node_id: e.id,
            span: e.span,
            parent: None,
        });
    }

    fn define_trait(&mut self, t: &TraitDef) {
        let def_id = self.symbols.fresh_def_id();

        let _ = self.symbols.define_type(t.name.clone(), def_id);

        self.symbols.insert(Symbol {
            def_id,
            name: t.name.clone(),
            kind: DefKind::Trait,
            node_id: t.id,
            span: t.span,
            parent: None,
        });
    }

    fn define_global(&mut self, g: &GlobalDef) {
        let def_id = self.symbols.fresh_def_id();

        if let Pattern::Binding { name, mutable } = &g.pattern {
            let _ = self.symbols.define(name.clone(), def_id);
            self.symbols.insert(Symbol {
                def_id,
                name: name.clone(),
                kind: if g.is_const {
                    DefKind::Const
                } else {
                    DefKind::Variable { mutable: *mutable }
                },
                node_id: g.id,
                span: g.span,
                parent: None,
            });
        }
    }

    /// Second pass: resolve bodies
    fn resolve_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.resolve_function(f),
            Item::Struct(s) => self.resolve_struct(s),
            Item::Enum(e) => self.resolve_enum(e),
            Item::TypeAlias(t) => self.resolve_type_alias(t),
            Item::Global(g) => self.resolve_global(g),
            Item::Module(m) => self.resolve_module(m),
            Item::Import(_) => {} // Imports are fully handled in collect phase
            _ => {}
        }
    }

    fn resolve_module(&mut self, m: &ModuleDef) {
        // Enter the module
        let parent_path = self.symbols.current_module().path.clone();
        let mut module_path = parent_path;
        module_path.push(m.name.clone());
        let module_id = ModuleId::new(module_path);
        self.symbols.enter_module(module_id);

        // Resolve items if inline module
        if let Some(ref items) = m.items {
            for item in items {
                self.resolve_item(item);
            }
        }

        // Exit module
        self.symbols.exit_module();
    }

    fn resolve_function(&mut self, f: &FnDef) {
        let fn_def_id = self.symbols.def_for_node(f.id);
        self.symbols.push_scope(ScopeKind::Function, fn_def_id);

        // Resolve generic parameters
        for param in &f.generics.params {
            if let GenericParam::Type { name, bounds, .. } = param {
                let def_id = self.symbols.fresh_def_id();
                let _ = self.symbols.define_type(name.clone(), def_id);
                self.symbols.insert(Symbol {
                    def_id,
                    name: name.clone(),
                    kind: DefKind::TypeParam,
                    node_id: NodeId(0), // No node ID for generic params
                    span: Span::default(),
                    parent: fn_def_id,
                });
                for bound in bounds {
                    self.resolve_path_as_type(bound);
                }
            }
        }

        // Resolve parameters
        for param in &f.params {
            self.resolve_param(param);
        }

        // Resolve return type
        if let Some(ref ret_ty) = f.return_type {
            self.resolve_type_expr(ret_ty);
        }

        // Resolve effects
        for effect in &f.effects {
            self.resolve_effect_ref(effect);
        }

        // Resolve body
        self.resolve_block(&f.body);

        self.symbols.pop_scope();
    }

    fn resolve_struct(&mut self, s: &StructDef) {
        self.symbols.push_scope(ScopeKind::TypeDef, None);

        // Resolve generic parameters
        for param in &s.generics.params {
            if let GenericParam::Type { name, .. } = param {
                let def_id = self.symbols.fresh_def_id();
                let _ = self.symbols.define_type(name.clone(), def_id);
                self.symbols.insert(Symbol {
                    def_id,
                    name: name.clone(),
                    kind: DefKind::TypeParam,
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                });
            }
        }

        // Resolve field types
        for field in &s.fields {
            self.resolve_type_expr(&field.ty);
        }

        self.symbols.pop_scope();
    }

    fn resolve_enum(&mut self, e: &EnumDef) {
        self.symbols.push_scope(ScopeKind::TypeDef, None);

        // Resolve generic parameters
        for param in &e.generics.params {
            if let GenericParam::Type { name, .. } = param {
                let def_id = self.symbols.fresh_def_id();
                let _ = self.symbols.define_type(name.clone(), def_id);
            }
        }

        // Resolve variant types
        for variant in &e.variants {
            match &variant.data {
                VariantData::Tuple(types) => {
                    for ty in types {
                        self.resolve_type_expr(ty);
                    }
                }
                VariantData::Struct(fields) => {
                    for field in fields {
                        self.resolve_type_expr(&field.ty);
                    }
                }
                VariantData::Unit => {}
            }
        }

        self.symbols.pop_scope();
    }

    fn resolve_type_alias(&mut self, t: &TypeAliasDef) {
        self.symbols.push_scope(ScopeKind::TypeDef, None);

        // Resolve generic parameters
        for param in &t.generics.params {
            if let GenericParam::Type { name, .. } = param {
                let def_id = self.symbols.fresh_def_id();
                let _ = self.symbols.define_type(name.clone(), def_id);
            }
        }

        self.resolve_type_expr(&t.ty);

        self.symbols.pop_scope();
    }

    fn resolve_global(&mut self, g: &GlobalDef) {
        if let Some(ref ty) = g.ty {
            self.resolve_type_expr(ty);
        }
        self.resolve_expr(&g.value);
    }

    fn resolve_param(&mut self, param: &Param) {
        // First resolve the type
        self.resolve_type_expr(&param.ty);

        // Then bind the parameter name
        if let Pattern::Binding { name, mutable } = &param.pattern {
            let def_id = self.symbols.fresh_def_id();
            let _ = self.symbols.define(name.clone(), def_id);
            self.symbols.insert(Symbol {
                def_id,
                name: name.clone(),
                kind: DefKind::Parameter { mutable: *mutable },
                node_id: param.id,
                span: Span::default(),
                parent: None,
            });
        }
    }

    fn resolve_type_expr(&mut self, ty: &TypeExpr) {
        match ty {
            TypeExpr::Named { path, args, .. } => {
                self.resolve_path_as_type(path);
                for arg in args {
                    self.resolve_type_expr(arg);
                }
            }
            TypeExpr::Reference { inner, .. } => {
                self.resolve_type_expr(inner);
            }
            TypeExpr::RawPointer { inner, .. } => {
                self.resolve_type_expr(inner);
            }
            TypeExpr::Array { element, .. } => {
                self.resolve_type_expr(element);
            }
            TypeExpr::Tuple(types) => {
                for t in types {
                    self.resolve_type_expr(t);
                }
            }
            TypeExpr::Function {
                params,
                return_type,
                effects,
            } => {
                for p in params {
                    self.resolve_type_expr(p);
                }
                self.resolve_type_expr(return_type);
                for eff in effects {
                    self.resolve_effect_ref(eff);
                }
            }
            TypeExpr::Unit | TypeExpr::SelfType | TypeExpr::Infer => {}

            // Epistemic types
            TypeExpr::Knowledge { value_type, .. } => {
                self.resolve_type_expr(value_type);
            }
            TypeExpr::Quantity { numeric_type, .. } => {
                self.resolve_type_expr(numeric_type);
            }
            TypeExpr::Tensor { element_type, .. } => {
                self.resolve_type_expr(element_type);
            }
            TypeExpr::Ontology { .. } => {}
            TypeExpr::Linear { inner, .. } => {
                self.resolve_type_expr(inner);
            }
            TypeExpr::Effected { inner, .. } => {
                self.resolve_type_expr(inner);
            }
            TypeExpr::Tile { element_type, .. } => {
                self.resolve_type_expr(element_type);
            }
            TypeExpr::Refinement { base_type, .. } => {
                self.resolve_type_expr(base_type);
            }
        }
    }

    fn resolve_path_as_type(&mut self, path: &Path) {
        if path.is_simple() {
            let name = path.name().unwrap();
            if self.symbols.lookup_type(name).is_none() {
                // Find similar type names for suggestion
                let suggestion = self.find_similar_type(name);
                self.errors.push(ResolveError::UndefinedType {
                    name: name.to_string(),
                    suggestion,
                    span: SourceSpan::from(0..1),
                });
            }
        }
        // TODO: multi-segment paths
    }

    /// Find a similar type name using Levenshtein distance
    fn find_similar_type(&self, name: &str) -> Option<String> {
        let type_names = self.symbols.all_type_names();
        find_similar_name(name, &type_names)
    }

    fn resolve_effect_ref(&mut self, effect: &EffectRef) {
        self.resolve_path_as_type(&effect.name);
        for arg in &effect.args {
            self.resolve_type_expr(arg);
        }
    }

    fn resolve_block(&mut self, block: &Block) {
        self.symbols.push_scope(ScopeKind::Block, None);

        for stmt in &block.stmts {
            self.resolve_stmt(stmt);
        }

        self.symbols.pop_scope();
    }

    fn resolve_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                is_mut,
                pattern,
                ty,
                value,
            } => {
                // Resolve initializer first (before binding)
                if let Some(init) = value {
                    self.resolve_expr(init);
                }
                if let Some(t) = ty {
                    self.resolve_type_expr(t);
                }

                // Now bind the variable
                self.resolve_pattern(pattern, *is_mut);
            }
            Stmt::Expr { expr, .. } => {
                self.resolve_expr(expr);
            }
            Stmt::Assign { target, value, .. } => {
                self.resolve_expr(target);
                self.resolve_expr(value);
            }
            Stmt::Empty | Stmt::MacroInvocation(_) => {}
        }
    }

    fn resolve_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal { .. } => {}

            Expr::Path { id, path } => {
                if path.is_simple() {
                    let name = path.name().unwrap();
                    if let Some(def_id) = self.symbols.lookup(name) {
                        self.symbols.record_ref(*id, def_id);
                    } else {
                        self.errors.push(ResolveError::UndefinedVar {
                            name: name.to_string(),
                            span: SourceSpan::from(0..1),
                        });
                    }
                }
                // TODO: multi-segment paths
            }

            Expr::Binary { left, right, .. } => {
                self.resolve_expr(left);
                self.resolve_expr(right);
            }

            Expr::Unary { expr, .. } => {
                self.resolve_expr(expr);
            }

            Expr::Call { callee, args, .. } => {
                self.resolve_expr(callee);
                for arg in args {
                    self.resolve_expr(arg);
                }
            }

            Expr::MethodCall { receiver, args, .. } => {
                self.resolve_expr(receiver);
                for arg in args {
                    self.resolve_expr(arg);
                }
            }

            Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
                self.resolve_expr(base);
            }

            Expr::Index { base, index, .. } => {
                self.resolve_expr(base);
                self.resolve_expr(index);
            }

            Expr::Cast { expr, ty, .. } => {
                self.resolve_expr(expr);
                self.resolve_type_expr(ty);
            }

            Expr::Block { block, .. } => {
                self.resolve_block(block);
            }

            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.resolve_expr(condition);
                self.resolve_block(then_branch);
                if let Some(else_expr) = else_branch {
                    self.resolve_expr(else_expr);
                }
            }

            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.resolve_expr(scrutinee);
                for arm in arms {
                    self.symbols.push_scope(ScopeKind::Block, None);
                    self.resolve_pattern(&arm.pattern, false);
                    if let Some(guard) = &arm.guard {
                        self.resolve_expr(guard);
                    }
                    self.resolve_expr(&arm.body);
                    self.symbols.pop_scope();
                }
            }

            Expr::Loop { body, .. } => {
                self.resolve_block(body);
            }

            Expr::While {
                condition, body, ..
            } => {
                self.resolve_expr(condition);
                self.resolve_block(body);
            }

            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => {
                self.resolve_expr(iter);
                self.symbols.push_scope(ScopeKind::Block, None);
                self.resolve_pattern(pattern, false);
                self.resolve_block(body);
                self.symbols.pop_scope();
            }

            Expr::Return { value, .. } => {
                if let Some(val) = value {
                    self.resolve_expr(val);
                }
            }

            Expr::Break { value, .. } => {
                if let Some(val) = value {
                    self.resolve_expr(val);
                }
            }

            Expr::Continue { .. } => {}

            Expr::Closure {
                params,
                return_type,
                body,
                ..
            } => {
                self.symbols.push_scope(ScopeKind::Function, None);
                for (name, ty) in params {
                    let def_id = self.symbols.fresh_def_id();
                    let _ = self.symbols.define(name.clone(), def_id);
                    if let Some(t) = ty {
                        self.resolve_type_expr(t);
                    }
                }
                if let Some(ret) = return_type {
                    self.resolve_type_expr(ret);
                }
                self.resolve_expr(body);
                self.symbols.pop_scope();
            }

            Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
                for elem in elements {
                    self.resolve_expr(elem);
                }
            }

            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.resolve_expr(s);
                }
                if let Some(e) = end {
                    self.resolve_expr(e);
                }
            }

            Expr::StructLit { path, fields, .. } => {
                self.resolve_path_as_type(path);
                for (_, value) in fields {
                    self.resolve_expr(value);
                }
            }

            Expr::Try { expr, .. } | Expr::Await { expr, .. } | Expr::Spawn { expr, .. } => {
                self.resolve_expr(expr);
            }

            Expr::Perform { effect, args, .. } => {
                self.resolve_path_as_type(effect);
                for arg in args {
                    self.resolve_expr(arg);
                }
            }

            Expr::Handle { expr, handler, .. } => {
                self.resolve_expr(expr);
                self.resolve_path_as_type(handler);
            }

            Expr::Sample { distribution, .. } => {
                self.resolve_expr(distribution);
            }

            Expr::AsyncBlock { block, .. } => {
                self.resolve_block(block);
            }

            Expr::AsyncClosure {
                params,
                return_type,
                body,
                ..
            } => {
                self.symbols.push_scope(ScopeKind::Function, None);
                for (name, ty) in params {
                    let def_id = self.symbols.fresh_def_id();
                    let _ = self.symbols.define(name.clone(), def_id);
                    if let Some(t) = ty {
                        self.resolve_type_expr(t);
                    }
                }
                if let Some(ret) = return_type {
                    self.resolve_type_expr(ret);
                }
                self.resolve_expr(body);
                self.symbols.pop_scope();
            }

            Expr::Select { arms, .. } => {
                for arm in arms {
                    self.resolve_expr(&arm.future);
                    self.symbols.push_scope(ScopeKind::Block, None);
                    self.resolve_pattern(&arm.pattern, false);
                    if let Some(guard) = &arm.guard {
                        self.resolve_expr(guard);
                    }
                    self.resolve_expr(&arm.body);
                    self.symbols.pop_scope();
                }
            }

            Expr::Join { futures, .. } => {
                for future in futures {
                    self.resolve_expr(future);
                }
            }

            Expr::MacroInvocation(_) => {}

            // Epistemic expressions
            Expr::Do { interventions, .. } => {
                for (_, value) in interventions {
                    self.resolve_expr(value);
                }
            }

            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                self.resolve_expr(factual);
                self.resolve_expr(intervention);
                self.resolve_expr(outcome);
            }

            Expr::KnowledgeExpr {
                value,
                epsilon,
                validity,
                provenance,
                ..
            } => {
                self.resolve_expr(value);
                if let Some(e) = epsilon {
                    self.resolve_expr(e);
                }
                if let Some(v) = validity {
                    self.resolve_expr(v);
                }
                if let Some(p) = provenance {
                    self.resolve_expr(p);
                }
            }

            Expr::Uncertain {
                value, uncertainty, ..
            } => {
                self.resolve_expr(value);
                self.resolve_expr(uncertainty);
            }

            Expr::GpuAnnotated { expr, .. } => {
                self.resolve_expr(expr);
            }

            Expr::Observe {
                data, distribution, ..
            } => {
                self.resolve_expr(data);
                self.resolve_expr(distribution);
            }

            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                self.resolve_expr(target);
                for g in given {
                    self.resolve_expr(g);
                }
                for (_, value) in interventions {
                    self.resolve_expr(value);
                }
            }

            // Ontology terms are literals, no resolution needed
            Expr::OntologyTerm { .. } => {}
        }
    }

    fn resolve_pattern(&mut self, pat: &Pattern, is_mut: bool) {
        match pat {
            Pattern::Wildcard | Pattern::Literal(_) => {}

            Pattern::Binding { name, mutable } => {
                let def_id = self.symbols.fresh_def_id();
                let _ = self.symbols.define(name.clone(), def_id);
                self.symbols.insert(Symbol {
                    def_id,
                    name: name.clone(),
                    kind: DefKind::Variable {
                        mutable: is_mut || *mutable,
                    },
                    node_id: NodeId(0),
                    span: Span::default(),
                    parent: None,
                });
            }

            Pattern::Tuple(patterns) => {
                for p in patterns {
                    self.resolve_pattern(p, is_mut);
                }
            }

            Pattern::Struct { path, fields } => {
                self.resolve_path_as_type(path);
                for (_, pattern) in fields {
                    self.resolve_pattern(pattern, is_mut);
                }
            }

            Pattern::Enum { path, patterns } => {
                // Resolve the enum variant path
                if path.segments.len() >= 2 {
                    // Full path like Option::Some
                    let type_name = &path.segments[0];
                    if self.symbols.lookup_type(type_name).is_none() {
                        let suggestion = self.find_similar_type(type_name);
                        self.errors.push(ResolveError::UndefinedType {
                            name: type_name.clone(),
                            suggestion,
                            span: SourceSpan::from(0..1),
                        });
                    }
                }
                if let Some(pats) = patterns {
                    for p in pats {
                        self.resolve_pattern(p, is_mut);
                    }
                }
            }

            Pattern::Or(patterns) => {
                for p in patterns {
                    self.resolve_pattern(p, is_mut);
                }
            }
        }
    }

    fn span_to_source(&self, span: Span) -> SourceSpan {
        SourceSpan::from(span.start..span.end)
    }
}

impl Default for Resolver {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute Levenshtein distance between two strings
fn levenshtein_distance(a: &str, b: &str) -> usize {
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

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Find a similar name from a list of candidates
fn find_similar_name(target: &str, candidates: &[String]) -> Option<String> {
    let max_distance = match target.len() {
        0..=2 => 0, // Exact match only for very short names
        3..=5 => 1, // Allow 1 edit for short names
        _ => 2,     // Allow 2 edits for longer names
    };

    candidates
        .iter()
        .filter_map(|candidate| {
            let dist = levenshtein_distance(target, candidate);
            if dist <= max_distance && dist > 0 {
                Some((candidate.clone(), dist))
            } else {
                None
            }
        })
        .min_by_key(|(_, dist)| *dist)
        .map(|(name, _)| name)
}
