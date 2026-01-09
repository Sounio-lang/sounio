//! Find references provider
//!
//! Provides functionality to find all references to a symbol by walking the AST
//! and building an index of symbol usage locations.

use std::collections::HashMap;
use tower_lsp::lsp_types::*;

use crate::ast::{
    Ast, Block, CausalModelDef, EffectDef, EnumDef, Expr, FnDef, GlobalDef, HandlerDef, ImplDef,
    ImplItem, Item, MatchArm, OdeDef, Param, Pattern, PdeDef, SelectArm, Stmt, StructDef,
    TraitDef, TraitItem, TypeAliasDef, TypeExpr, VariantData,
};
use crate::common::{NodeId, Span};
use crate::lsp::document::LineIndex;
use crate::resolve::{DefId, SymbolTable};

/// Kind of reference usage
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceKind {
    /// The definition site of a symbol
    Definition,
    /// A read/use of a symbol
    Read,
    /// A write/assignment to a symbol
    Write,
    /// An import of a symbol
    Import,
    /// Usage in a type annotation
    TypeAnnotation,
}

/// A single reference location
#[derive(Debug, Clone)]
pub struct ReferenceLocation {
    /// URI of the file containing the reference
    pub uri: Url,
    /// Range in the document
    pub range: Range,
    /// Kind of reference
    pub kind: ReferenceKind,
    /// The referenced symbol's DefId
    pub def_id: DefId,
}

/// Index of all references in a file or workspace
#[derive(Debug, Default)]
pub struct ReferenceIndex {
    /// DefId -> list of reference locations
    references: HashMap<DefId, Vec<ReferenceLocation>>,
    /// NodeId -> DefId mapping for quick lookup
    node_refs: HashMap<NodeId, DefId>,
    /// Name -> list of reference locations (for symbols not yet resolved)
    name_refs: HashMap<String, Vec<ReferenceLocation>>,
}

impl ReferenceIndex {
    /// Create a new empty reference index
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a reference index from an AST and symbol table
    pub fn build(ast: &Ast, symbols: &SymbolTable, uri: &Url, line_index: &LineIndex) -> Self {
        let mut index = Self::new();
        let mut walker = ReferenceWalker::new(&mut index, symbols, uri, line_index);
        walker.walk_ast(ast);
        index
    }

    /// Add a reference
    pub fn add_reference(&mut self, def_id: DefId, location: ReferenceLocation) {
        self.references.entry(def_id).or_default().push(location);
    }

    /// Add a name-based reference (for unresolved symbols)
    pub fn add_name_reference(&mut self, name: String, location: ReferenceLocation) {
        self.name_refs.entry(name).or_default().push(location);
    }

    /// Record a NodeId -> DefId mapping
    pub fn record_node_ref(&mut self, node_id: NodeId, def_id: DefId) {
        self.node_refs.insert(node_id, def_id);
    }

    /// Get all references to a DefId
    pub fn get_references(&self, def_id: DefId) -> Vec<&ReferenceLocation> {
        self.references
            .get(&def_id)
            .map(|refs| refs.iter().collect())
            .unwrap_or_default()
    }

    /// Get all references by name (useful when DefId is not available)
    pub fn get_references_by_name(&self, name: &str) -> Vec<&ReferenceLocation> {
        self.name_refs
            .get(name)
            .map(|refs| refs.iter().collect())
            .unwrap_or_default()
    }

    /// Get DefId for a NodeId
    pub fn get_def_for_node(&self, node_id: NodeId) -> Option<DefId> {
        self.node_refs.get(&node_id).copied()
    }

    /// Convert to LSP Locations
    pub fn to_locations(&self, def_id: DefId) -> Vec<Location> {
        self.get_references(def_id)
            .into_iter()
            .map(|r| Location {
                uri: r.uri.clone(),
                range: r.range,
            })
            .collect()
    }

    /// Convert name references to LSP Locations
    pub fn name_to_locations(&self, name: &str) -> Vec<Location> {
        self.get_references_by_name(name)
            .into_iter()
            .map(|r| Location {
                uri: r.uri.clone(),
                range: r.range,
            })
            .collect()
    }

    /// Merge another index into this one
    pub fn merge(&mut self, other: ReferenceIndex) {
        for (def_id, refs) in other.references {
            self.references.entry(def_id).or_default().extend(refs);
        }
        for (name, refs) in other.name_refs {
            self.name_refs.entry(name).or_default().extend(refs);
        }
        self.node_refs.extend(other.node_refs);
    }
}

/// AST walker that collects all references
struct ReferenceWalker<'a> {
    index: &'a mut ReferenceIndex,
    symbols: &'a SymbolTable,
    uri: &'a Url,
    line_index: &'a LineIndex,
}

impl<'a> ReferenceWalker<'a> {
    fn new(
        index: &'a mut ReferenceIndex,
        symbols: &'a SymbolTable,
        uri: &'a Url,
        line_index: &'a LineIndex,
    ) -> Self {
        Self {
            index,
            symbols,
            uri,
            line_index,
        }
    }

    /// Convert a span to an LSP Range
    fn span_to_range(&self, span: &Span) -> Range {
        self.line_index.span_to_range(span)
    }

    /// Create a reference location
    fn make_ref(&self, span: &Span, kind: ReferenceKind, def_id: DefId) -> ReferenceLocation {
        ReferenceLocation {
            uri: self.uri.clone(),
            range: self.span_to_range(span),
            kind,
            def_id,
        }
    }

    /// Record a reference by name (when DefId lookup is needed)
    fn record_name_ref(&mut self, name: &str, span: &Span, kind: ReferenceKind) {
        // Try to resolve the name to a DefId
        if let Some(def_id) = self.symbols.lookup(name).or_else(|| self.symbols.lookup_type(name))
        {
            let location = self.make_ref(span, kind, def_id);
            self.index.add_reference(def_id, location);
        } else {
            // Store as name reference for later resolution
            let location = ReferenceLocation {
                uri: self.uri.clone(),
                range: self.span_to_range(span),
                kind,
                def_id: DefId::new(0), // Placeholder
            };
            self.index.add_name_reference(name.to_string(), location);
        }
    }

    /// Walk the entire AST
    fn walk_ast(&mut self, ast: &Ast) {
        for item in &ast.items {
            self.walk_item(item);
        }
    }

    /// Walk a top-level item
    fn walk_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.walk_function(f),
            Item::Struct(s) => self.walk_struct(s),
            Item::Enum(e) => self.walk_enum(e),
            Item::Trait(t) => self.walk_trait(t),
            Item::Impl(i) => self.walk_impl(i),
            Item::TypeAlias(t) => self.walk_type_alias(t),
            Item::Effect(e) => self.walk_effect(e),
            Item::Handler(h) => self.walk_handler(h),
            Item::Import(i) => {
                // Record import references
                for seg in &i.path.segments {
                    // Create a span for the segment (approximate)
                    self.record_name_ref(seg, &i.span, ReferenceKind::Import);
                }
                if let Some(items) = &i.items {
                    for item in items {
                        if !item.is_glob {
                            self.record_name_ref(&item.name, &i.span, ReferenceKind::Import);
                        }
                    }
                }
            }
            Item::Export(e) => {
                for name in &e.names {
                    self.record_name_ref(name, &e.span, ReferenceKind::Read);
                }
            }
            Item::Global(g) => self.walk_global(g),
            Item::OdeDef(o) => self.walk_ode(o),
            Item::PdeDef(p) => self.walk_pde(p),
            Item::CausalModel(c) => self.walk_causal_model(c),
            Item::Module(m) => {
                if let Some(items) = &m.items {
                    for item in items {
                        self.walk_item(item);
                    }
                }
            }
            Item::Extern(e) => {
                for item in &e.items {
                    match item {
                        crate::ast::ExternItem::Fn(f) => {
                            for param in &f.params {
                                self.walk_type(&param.ty);
                            }
                            if let Some(ret) = &f.return_type {
                                self.walk_type(ret);
                            }
                        }
                        crate::ast::ExternItem::Static(s) => {
                            self.walk_type(&s.ty);
                        }
                        crate::ast::ExternItem::Type(_) => {}
                    }
                }
            }
            Item::MacroInvocation(_) => {}
            Item::OntologyImport(_) => {}
            Item::AlignDecl(_) => {}
        }
    }

    /// Walk a function definition
    fn walk_function(&mut self, f: &FnDef) {
        // Record the function definition itself
        self.record_name_ref(&f.name, &f.span, ReferenceKind::Definition);

        // Walk generic parameters
        for param in &f.generics.params {
            match param {
                crate::ast::GenericParam::Type { bounds, default, .. } => {
                    for bound in bounds {
                        for seg in &bound.segments {
                            self.record_name_ref(seg, &f.span, ReferenceKind::TypeAnnotation);
                        }
                    }
                    if let Some(default_ty) = default {
                        self.walk_type(default_ty);
                    }
                }
                crate::ast::GenericParam::Const { ty, .. } => {
                    self.walk_type(ty);
                }
                crate::ast::GenericParam::Effect { .. } => {}
            }
        }

        // Walk parameters
        for param in &f.params {
            self.walk_param(param);
        }

        // Walk return type
        if let Some(ret) = &f.return_type {
            self.walk_type(ret);
        }

        // Walk effects
        for effect in &f.effects {
            for seg in &effect.name.segments {
                self.record_name_ref(seg, &f.span, ReferenceKind::TypeAnnotation);
            }
        }

        // Walk where clause
        for pred in &f.where_clause {
            self.walk_type(&pred.ty);
            for bound in &pred.bounds {
                for seg in &bound.segments {
                    self.record_name_ref(seg, &f.span, ReferenceKind::TypeAnnotation);
                }
            }
        }

        // Walk body
        self.walk_block(&f.body);
    }

    /// Walk a parameter
    fn walk_param(&mut self, param: &Param) {
        self.walk_pattern(&param.pattern);
        self.walk_type(&param.ty);
    }

    /// Walk a struct definition
    fn walk_struct(&mut self, s: &StructDef) {
        // Record the struct definition
        self.record_name_ref(&s.name, &s.span, ReferenceKind::Definition);

        // Walk generic parameters
        for param in &s.generics.params {
            if let crate::ast::GenericParam::Type { bounds, default, .. } = param {
                for bound in bounds {
                    for seg in &bound.segments {
                        self.record_name_ref(seg, &s.span, ReferenceKind::TypeAnnotation);
                    }
                }
                if let Some(default_ty) = default {
                    self.walk_type(default_ty);
                }
            }
        }

        // Walk fields
        for field in &s.fields {
            self.walk_type(&field.ty);
        }
    }

    /// Walk an enum definition
    fn walk_enum(&mut self, e: &EnumDef) {
        // Record the enum definition
        self.record_name_ref(&e.name, &e.span, ReferenceKind::Definition);

        // Walk generic parameters
        for param in &e.generics.params {
            if let crate::ast::GenericParam::Type { bounds, default, .. } = param {
                for bound in bounds {
                    for seg in &bound.segments {
                        self.record_name_ref(seg, &e.span, ReferenceKind::TypeAnnotation);
                    }
                }
                if let Some(default_ty) = default {
                    self.walk_type(default_ty);
                }
            }
        }

        // Walk variants
        for variant in &e.variants {
            match &variant.data {
                VariantData::Unit => {}
                VariantData::Tuple(types) => {
                    for ty in types {
                        self.walk_type(ty);
                    }
                }
                VariantData::Struct(fields) => {
                    for field in fields {
                        self.walk_type(&field.ty);
                    }
                }
            }
        }
    }

    /// Walk a trait definition
    fn walk_trait(&mut self, t: &TraitDef) {
        // Record the trait definition
        self.record_name_ref(&t.name, &t.span, ReferenceKind::Definition);

        // Walk supertraits
        for supertrait in &t.supertraits {
            for seg in &supertrait.segments {
                self.record_name_ref(seg, &t.span, ReferenceKind::TypeAnnotation);
            }
        }

        // Walk items
        for item in &t.items {
            match item {
                TraitItem::Fn(f) => {
                    for param in &f.params {
                        self.walk_param(param);
                    }
                    if let Some(ret) = &f.return_type {
                        self.walk_type(ret);
                    }
                    if let Some(body) = &f.default_body {
                        self.walk_block(body);
                    }
                }
                TraitItem::Type(t) => {
                    for bound in &t.bounds {
                        for seg in &bound.segments {
                            self.record_name_ref(
                                seg,
                                &Span::dummy(),
                                ReferenceKind::TypeAnnotation,
                            );
                        }
                    }
                    if let Some(default) = &t.default {
                        self.walk_type(default);
                    }
                }
            }
        }
    }

    /// Walk an impl block
    fn walk_impl(&mut self, i: &ImplDef) {
        // Walk trait reference
        if let Some(trait_ref) = &i.trait_ref {
            for seg in &trait_ref.segments {
                self.record_name_ref(seg, &i.span, ReferenceKind::TypeAnnotation);
            }
        }

        // Walk target type
        self.walk_type(&i.target_type);

        // Walk generic parameters
        for param in &i.generics.params {
            if let crate::ast::GenericParam::Type { bounds, default, .. } = param {
                for bound in bounds {
                    for seg in &bound.segments {
                        self.record_name_ref(seg, &i.span, ReferenceKind::TypeAnnotation);
                    }
                }
                if let Some(default_ty) = default {
                    self.walk_type(default_ty);
                }
            }
        }

        // Walk items
        for item in &i.items {
            match item {
                ImplItem::Fn(f) => self.walk_function(f),
                ImplItem::Type(t) => {
                    self.walk_type(&t.ty);
                }
            }
        }
    }

    /// Walk a type alias
    fn walk_type_alias(&mut self, t: &TypeAliasDef) {
        self.record_name_ref(&t.name, &t.span, ReferenceKind::Definition);
        self.walk_type(&t.ty);
    }

    /// Walk an effect definition
    fn walk_effect(&mut self, e: &EffectDef) {
        self.record_name_ref(&e.name, &e.span, ReferenceKind::Definition);

        for op in &e.operations {
            for param in &op.params {
                self.walk_param(param);
            }
            if let Some(ret) = &op.return_type {
                self.walk_type(ret);
            }
        }
    }

    /// Walk a handler definition
    fn walk_handler(&mut self, h: &HandlerDef) {
        self.record_name_ref(&h.name, &h.span, ReferenceKind::Definition);

        for seg in &h.effect.segments {
            self.record_name_ref(seg, &h.span, ReferenceKind::TypeAnnotation);
        }

        for case in &h.cases {
            for param in &case.params {
                self.walk_param(param);
            }
            self.walk_expr(&case.body);
        }
    }

    /// Walk a global definition
    fn walk_global(&mut self, g: &GlobalDef) {
        self.walk_pattern(&g.pattern);
        if let Some(ty) = &g.ty {
            self.walk_type(ty);
        }
        self.walk_expr(&g.value);
    }

    /// Walk an ODE definition
    fn walk_ode(&mut self, o: &OdeDef) {
        self.record_name_ref(&o.name, &o.span, ReferenceKind::Definition);

        for param in &o.params {
            self.walk_type(&param.ty);
            if let Some(default) = &param.default {
                self.walk_expr(default);
            }
        }

        for state in &o.state {
            self.walk_type(&state.ty);
        }

        for eq in &o.equations {
            self.walk_expr(&eq.rhs);
        }
    }

    /// Walk a PDE definition
    fn walk_pde(&mut self, p: &PdeDef) {
        self.record_name_ref(&p.name, &p.span, ReferenceKind::Definition);

        for param in &p.params {
            self.walk_type(&param.ty);
            if let Some(default) = &param.default {
                self.walk_expr(default);
            }
        }

        for dim in &p.domain.dimensions {
            self.walk_expr(&dim.min);
            self.walk_expr(&dim.max);
        }

        self.walk_expr(&p.equation.rhs);

        if let Some(init) = &p.initial_condition {
            self.walk_expr(init);
        }

        for bc in &p.boundary_conditions {
            self.walk_expr(&bc.boundary.value);
            match &bc.condition {
                crate::ast::BoundaryConditionType::Dirichlet(e) => self.walk_expr(e),
                crate::ast::BoundaryConditionType::Neumann(e) => self.walk_expr(e),
                crate::ast::BoundaryConditionType::Robin { a, b, value } => {
                    self.walk_expr(a);
                    self.walk_expr(b);
                    self.walk_expr(value);
                }
                crate::ast::BoundaryConditionType::Periodic => {}
            }
        }
    }

    /// Walk a causal model definition
    fn walk_causal_model(&mut self, c: &CausalModelDef) {
        self.record_name_ref(&c.name, &c.span, ReferenceKind::Definition);

        for node in &c.nodes {
            if let Some(ty) = &node.ty {
                self.walk_type(ty);
            }
        }

        for eq in &c.equations {
            self.walk_expr(&eq.rhs);
        }
    }

    /// Walk a type expression
    fn walk_type(&mut self, ty: &TypeExpr) {
        match ty {
            TypeExpr::Unit | TypeExpr::SelfType | TypeExpr::Infer => {}
            TypeExpr::Named { path, args, .. } => {
                // Record references to the type name
                for seg in &path.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::TypeAnnotation);
                }
                for arg in args {
                    self.walk_type(arg);
                }
            }
            TypeExpr::Reference { inner, .. } | TypeExpr::RawPointer { inner, .. } => {
                self.walk_type(inner);
            }
            TypeExpr::Array { element, size } => {
                self.walk_type(element);
                if let Some(size_expr) = size {
                    self.walk_expr(size_expr);
                }
            }
            TypeExpr::Tuple(types) => {
                for t in types {
                    self.walk_type(t);
                }
            }
            TypeExpr::Function {
                params,
                return_type,
                effects,
            } => {
                for p in params {
                    self.walk_type(p);
                }
                self.walk_type(return_type);
                for effect in effects {
                    for seg in &effect.name.segments {
                        self.record_name_ref(seg, &Span::dummy(), ReferenceKind::TypeAnnotation);
                    }
                }
            }
            TypeExpr::Knowledge {
                value_type,
                epsilon,
                validity,
                provenance,
            } => {
                self.walk_type(value_type);
                if let Some(eps) = epsilon {
                    self.walk_expr(&eps.value);
                }
                if let Some(val) = validity {
                    self.walk_expr(&val.condition);
                }
                if let Some(prov) = provenance {
                    if let Some(src) = &prov.source {
                        self.walk_expr(src);
                    }
                }
            }
            TypeExpr::Quantity { numeric_type, .. } => {
                self.walk_type(numeric_type);
            }
            TypeExpr::Tensor {
                element_type,
                shape,
            } => {
                self.walk_type(element_type);
                for dim in shape {
                    if let crate::ast::TensorDim::Expr(e) = dim {
                        self.walk_expr(e);
                    }
                }
            }
            TypeExpr::Tile { element_type, .. } => {
                self.walk_type(element_type);
            }
            TypeExpr::Ontology { .. } => {}
            TypeExpr::Linear { inner, .. } => {
                self.walk_type(inner);
            }
            TypeExpr::Effected { inner, effects } => {
                self.walk_type(inner);
                for effect in &effects.effects {
                    self.record_name_ref(effect, &Span::dummy(), ReferenceKind::TypeAnnotation);
                }
            }
            TypeExpr::Refinement {
                base_type,
                predicate,
                ..
            } => {
                self.walk_type(base_type);
                self.walk_expr(predicate);
            }
        }
    }

    /// Walk a block
    fn walk_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.walk_stmt(stmt);
        }
    }

    /// Walk a statement
    fn walk_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let {
                pattern,
                ty,
                value,
                ..
            } => {
                self.walk_pattern(pattern);
                if let Some(ty) = ty {
                    self.walk_type(ty);
                }
                if let Some(value) = value {
                    self.walk_expr(value);
                }
            }
            Stmt::Expr { expr, .. } => {
                self.walk_expr(expr);
            }
            Stmt::Assign { target, value, .. } => {
                // Target is a write reference
                self.walk_expr_write(target);
                self.walk_expr(value);
            }
            Stmt::Empty => {}
            Stmt::MacroInvocation(_) => {}
        }
    }

    /// Walk an expression that is being written to (assignment target)
    fn walk_expr_write(&mut self, expr: &Expr) {
        match expr {
            Expr::Path { path, .. } => {
                if let Some(name) = path.segments.last() {
                    self.record_name_ref(name, &Span::dummy(), ReferenceKind::Write);
                }
            }
            Expr::Field { base, .. } => {
                self.walk_expr(base);
            }
            Expr::Index { base, index, .. } => {
                self.walk_expr(base);
                self.walk_expr(index);
            }
            Expr::Unary {
                op: crate::ast::UnaryOp::Deref,
                expr,
                ..
            } => {
                self.walk_expr(expr);
            }
            _ => self.walk_expr(expr),
        }
    }

    /// Walk an expression
    fn walk_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Literal { .. } => {}
            Expr::Path { path, .. } => {
                // Record references to identifiers
                for seg in &path.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::Read);
                }
            }
            Expr::Binary { left, right, .. } => {
                self.walk_expr(left);
                self.walk_expr(right);
            }
            Expr::Unary { expr, .. } => {
                self.walk_expr(expr);
            }
            Expr::Call { callee, args, .. } => {
                self.walk_expr(callee);
                for arg in args {
                    self.walk_expr(arg);
                }
            }
            Expr::MethodCall {
                receiver, args, ..
            } => {
                self.walk_expr(receiver);
                for arg in args {
                    self.walk_expr(arg);
                }
            }
            Expr::Field { base, .. } => {
                self.walk_expr(base);
            }
            Expr::TupleField { base, .. } => {
                self.walk_expr(base);
            }
            Expr::Index { base, index, .. } => {
                self.walk_expr(base);
                self.walk_expr(index);
            }
            Expr::Cast { expr, ty, .. } => {
                self.walk_expr(expr);
                self.walk_type(ty);
            }
            Expr::Block { block, .. } => {
                self.walk_block(block);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.walk_expr(condition);
                self.walk_block(then_branch);
                if let Some(else_expr) = else_branch {
                    self.walk_expr(else_expr);
                }
            }
            Expr::Match { scrutinee, arms, .. } => {
                self.walk_expr(scrutinee);
                for arm in arms {
                    self.walk_match_arm(arm);
                }
            }
            Expr::Loop { body, .. } => {
                self.walk_block(body);
            }
            Expr::While {
                condition, body, ..
            } => {
                self.walk_expr(condition);
                self.walk_block(body);
            }
            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => {
                self.walk_pattern(pattern);
                self.walk_expr(iter);
                self.walk_block(body);
            }
            Expr::Return { value, .. } => {
                if let Some(v) = value {
                    self.walk_expr(v);
                }
            }
            Expr::Break { value, .. } => {
                if let Some(v) = value {
                    self.walk_expr(v);
                }
            }
            Expr::Continue { .. } => {}
            Expr::Closure { body, .. } => {
                self.walk_expr(body);
            }
            Expr::Tuple { elements, .. } => {
                for e in elements {
                    self.walk_expr(e);
                }
            }
            Expr::Array { elements, .. } => {
                for e in elements {
                    self.walk_expr(e);
                }
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.walk_expr(s);
                }
                if let Some(e) = end {
                    self.walk_expr(e);
                }
            }
            Expr::StructLit { path, fields, .. } => {
                for seg in &path.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::Read);
                }
                for (_, field_expr) in fields {
                    self.walk_expr(field_expr);
                }
            }
            Expr::Try { expr, .. } => {
                self.walk_expr(expr);
            }
            Expr::Perform { effect, args, .. } => {
                for seg in &effect.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::Read);
                }
                for arg in args {
                    self.walk_expr(arg);
                }
            }
            Expr::Handle { expr, handler, .. } => {
                self.walk_expr(expr);
                for seg in &handler.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::Read);
                }
            }
            Expr::Sample { distribution, .. } => {
                self.walk_expr(distribution);
            }
            Expr::Await { expr, .. } => {
                self.walk_expr(expr);
            }
            Expr::AsyncBlock { block, .. } => {
                self.walk_block(block);
            }
            Expr::AsyncClosure { body, .. } => {
                self.walk_expr(body);
            }
            Expr::Spawn { expr, .. } => {
                self.walk_expr(expr);
            }
            Expr::Select { arms, .. } => {
                for arm in arms {
                    self.walk_select_arm(arm);
                }
            }
            Expr::Join { futures, .. } => {
                for f in futures {
                    self.walk_expr(f);
                }
            }
            Expr::MacroInvocation(_) => {}
            Expr::OntologyTerm { .. } => {}
            Expr::Do { interventions, .. } => {
                for (_, value) in interventions {
                    self.walk_expr(value);
                }
            }
            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => {
                self.walk_expr(factual);
                self.walk_expr(intervention);
                self.walk_expr(outcome);
            }
            Expr::KnowledgeExpr {
                value,
                epsilon,
                validity,
                provenance,
                ..
            } => {
                self.walk_expr(value);
                if let Some(e) = epsilon {
                    self.walk_expr(e);
                }
                if let Some(v) = validity {
                    self.walk_expr(v);
                }
                if let Some(p) = provenance {
                    self.walk_expr(p);
                }
            }
            Expr::Uncertain {
                value, uncertainty, ..
            } => {
                self.walk_expr(value);
                self.walk_expr(uncertainty);
            }
            Expr::GpuAnnotated { expr, .. } => {
                self.walk_expr(expr);
            }
            Expr::Observe {
                data, distribution, ..
            } => {
                self.walk_expr(data);
                self.walk_expr(distribution);
            }
            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                self.walk_expr(target);
                for g in given {
                    self.walk_expr(g);
                }
                for (_, value) in interventions {
                    self.walk_expr(value);
                }
            }
        }
    }

    /// Walk a match arm
    fn walk_match_arm(&mut self, arm: &MatchArm) {
        self.walk_pattern(&arm.pattern);
        if let Some(guard) = &arm.guard {
            self.walk_expr(guard);
        }
        self.walk_expr(&arm.body);
    }

    /// Walk a select arm
    fn walk_select_arm(&mut self, arm: &SelectArm) {
        self.walk_expr(&arm.future);
        self.walk_pattern(&arm.pattern);
        if let Some(guard) = &arm.guard {
            self.walk_expr(guard);
        }
        self.walk_expr(&arm.body);
    }

    /// Walk a pattern
    fn walk_pattern(&mut self, pattern: &Pattern) {
        match pattern {
            Pattern::Wildcard => {}
            Pattern::Literal(_) => {}
            Pattern::Binding { name, .. } => {
                // Record the binding as a definition
                self.record_name_ref(name, &Span::dummy(), ReferenceKind::Definition);
            }
            Pattern::Tuple(patterns) => {
                for p in patterns {
                    self.walk_pattern(p);
                }
            }
            Pattern::Struct { path, fields, .. } => {
                for seg in &path.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::Read);
                }
                for (_, pattern) in fields {
                    self.walk_pattern(pattern);
                }
            }
            Pattern::Enum { path, patterns, .. } => {
                for seg in &path.segments {
                    self.record_name_ref(seg, &Span::dummy(), ReferenceKind::Read);
                }
                if let Some(patterns) = patterns {
                    for p in patterns {
                        self.walk_pattern(p);
                    }
                }
            }
            Pattern::Or(patterns) => {
                for p in patterns {
                    self.walk_pattern(p);
                }
            }
        }
    }
}

/// Provider for find-references
pub struct ReferencesProvider;

impl ReferencesProvider {
    /// Create a new references provider
    pub fn new() -> Self {
        Self
    }

    /// Find all references to a symbol in a single file using the AST
    pub fn find_references_in_ast(
        &self,
        name: &str,
        ast: &Ast,
        symbols: &SymbolTable,
        uri: &Url,
        line_index: &LineIndex,
    ) -> Vec<Location> {
        let index = ReferenceIndex::build(ast, symbols, uri, line_index);

        // First try to find by DefId
        if let Some(def_id) = symbols.lookup(name).or_else(|| symbols.lookup_type(name)) {
            let mut locations = index.to_locations(def_id);
            // Also include name-based references
            locations.extend(index.name_to_locations(name));
            // Deduplicate
            locations.sort_by(|a, b| {
                a.uri
                    .as_str()
                    .cmp(b.uri.as_str())
                    .then_with(|| a.range.start.line.cmp(&b.range.start.line))
                    .then_with(|| a.range.start.character.cmp(&b.range.start.character))
            });
            locations.dedup_by(|a, b| a.uri == b.uri && a.range == b.range);
            locations
        } else {
            // Fall back to name-based search
            index.name_to_locations(name)
        }
    }

    /// Find all references to a symbol (legacy method, uses definition only)
    pub fn find_references(
        &self,
        name: &str,
        symbols: &SymbolTable,
        current_uri: &Url,
    ) -> Vec<Location> {
        let mut locations = Vec::new();

        // Look up the definition first
        let def_id = symbols.lookup(name).or_else(|| symbols.lookup_type(name));

        if let Some(def_id) = def_id {
            if let Some(symbol) = symbols.get(def_id) {
                // Add the definition itself
                locations.push(self.span_to_location(&symbol.span, current_uri));
            }
        }

        locations
    }

    /// Convert a span to an LSP location
    fn span_to_location(&self, span: &Span, uri: &Url) -> Location {
        Location {
            uri: uri.clone(),
            range: Range {
                start: Position {
                    line: 0,
                    character: span.start as u32,
                },
                end: Position {
                    line: 0,
                    character: span.end as u32,
                },
            },
        }
    }
}

impl Default for ReferencesProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_kind_variants() {
        let kinds = [
            ReferenceKind::Definition,
            ReferenceKind::Read,
            ReferenceKind::Write,
            ReferenceKind::Import,
            ReferenceKind::TypeAnnotation,
        ];
        assert_eq!(kinds.len(), 5);
    }

    #[test]
    fn test_reference_index_new() {
        let index = ReferenceIndex::new();
        assert!(index.references.is_empty());
        assert!(index.name_refs.is_empty());
        assert!(index.node_refs.is_empty());
    }

    #[test]
    fn test_reference_index_add_name_reference() {
        let mut index = ReferenceIndex::new();
        let uri = Url::parse("file:///test.sio").unwrap();
        let location = ReferenceLocation {
            uri: uri.clone(),
            range: Range::default(),
            kind: ReferenceKind::Read,
            def_id: DefId::new(0),
        };

        index.add_name_reference("foo".to_string(), location);
        let refs = index.get_references_by_name("foo");
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn test_reference_index_merge() {
        let mut index1 = ReferenceIndex::new();
        let mut index2 = ReferenceIndex::new();

        let uri = Url::parse("file:///test.sio").unwrap();
        let location1 = ReferenceLocation {
            uri: uri.clone(),
            range: Range::default(),
            kind: ReferenceKind::Read,
            def_id: DefId::new(1),
        };
        let location2 = ReferenceLocation {
            uri: uri.clone(),
            range: Range::default(),
            kind: ReferenceKind::Write,
            def_id: DefId::new(1),
        };

        index1.add_reference(DefId::new(1), location1);
        index2.add_reference(DefId::new(1), location2);

        index1.merge(index2);
        let refs = index1.get_references(DefId::new(1));
        assert_eq!(refs.len(), 2);
    }
}
