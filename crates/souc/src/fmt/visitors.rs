//! AST visitor for code formatting
//!
//! This module provides a systematic visitor that converts AST nodes
//! to the Doc IR for pretty-printing.

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use super::config::FormatConfig;
use super::ir::Doc;
use crate::ast::{
    AssignOp, Ast, BinaryOp, Block, CausalModelDef, EffectDef, EnumDef, Expr, ExternBlock,
    ExternItem, FnDef, GenericParam, GlobalDef, HandlerDef, ImplDef, ImplItem, ImportDef, Item,
    Literal, MatchArm, ModuleDef, OdeDef, Param, Pattern, PdeDef, SelectArm, Stmt, StructDef,
    TraitDef, TraitItem, TypeAliasDef, TypeExpr, UnaryOp, VariantData, VariantDef, Visibility,
    WherePredicate,
};

/// AST visitor that generates Doc IR for formatting
pub struct FormatVisitor<'a> {
    /// Format configuration
    #[allow(dead_code)]
    config: &'a FormatConfig,
}

impl<'a> FormatVisitor<'a> {
    /// Create a new format visitor
    pub fn new(config: &'a FormatConfig) -> Self {
        Self { config }
    }

    // ==================== TOP-LEVEL ====================

    /// Visit the entire AST
    pub fn visit_ast(&self, ast: &Ast) -> Doc {
        let mut parts = Vec::new();

        for (i, item) in ast.items.iter().enumerate() {
            if i > 0 {
                // Add blank lines between items
                parts.push(Doc::Hardline);
                parts.push(Doc::Hardline);
            }
            parts.push(self.visit_item(item));
        }

        Doc::concat(parts)
    }

    /// Visit a top-level item
    pub fn visit_item(&self, item: &Item) -> Doc {
        match item {
            Item::Function(f) => self.visit_function(f),
            Item::Struct(s) => self.visit_struct(s),
            Item::Enum(e) => self.visit_enum(e),
            Item::Trait(t) => self.visit_trait(t),
            Item::Impl(i) => self.visit_impl(i),
            Item::TypeAlias(t) => self.visit_type_alias(t),
            Item::Effect(e) => self.visit_effect(e),
            Item::Handler(h) => self.visit_handler(h),
            Item::Import(u) => self.visit_import(u),
            Item::Export(e) => self.visit_export(e),
            Item::Extern(e) => self.visit_extern(e),
            Item::Global(g) => self.visit_global(g),
            Item::MacroInvocation(m) => Doc::text(format!("{}!(...)", m.name)),
            Item::OntologyImport(o) => self.visit_ontology_import(o),
            Item::AlignDecl(a) => self.visit_align(a),
            Item::OdeDef(o) => self.visit_ode(o),
            Item::PdeDef(p) => self.visit_pde(p),
            Item::CausalModel(c) => self.visit_causal_model(c),
            Item::Unit(u) => {
                let vis = if matches!(u.visibility, Visibility::Public) {
                    "pub "
                } else {
                    ""
                };
                if u.definition.is_some() {
                    Doc::text(format!("{}unit {} = ...;", vis, u.name))
                } else {
                    Doc::text(format!("{}unit {};", vis, u.name))
                }
            }
            Item::Module(m) => self.visit_module(m),
        }
    }

    // ==================== FUNCTIONS ====================

    /// Visit a function definition
    pub fn visit_function(&self, func: &FnDef) -> Doc {
        let mut parts = Vec::new();

        // Attributes
        for attr in &func.attributes {
            parts.push(Doc::text(format!("#[{}]", attr.name)));
            parts.push(Doc::Hardline);
        }

        // Visibility
        if matches!(func.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        // Modifiers
        if func.modifiers.is_async {
            parts.push(Doc::text("async "));
        }
        if func.modifiers.is_unsafe {
            parts.push(Doc::text("unsafe "));
        }
        if func.modifiers.is_kernel {
            parts.push(Doc::text("kernel "));
        }

        // ABI
        if let Some(abi) = &func.modifiers.abi {
            parts.push(Doc::text(format!("extern \"{}\" ", abi)));
        }

        // fn keyword and name
        parts.push(Doc::text(format!("fn {}", func.name)));

        // Generic parameters
        if !func.generics.params.is_empty() {
            parts.push(self.visit_generics(&func.generics.params));
        }

        // Parameters
        parts.push(self.visit_params(&func.params));

        // Return type
        if let Some(ret) = &func.return_type {
            parts.push(Doc::text(" -> "));
            parts.push(self.visit_type(ret));
        }

        // Effects
        if !func.effects.is_empty() {
            parts.push(Doc::text(" with "));
            let effects: Vec<Doc> = func
                .effects
                .iter()
                .map(|e| Doc::text(e.name.segments.join("::")))
                .collect();
            parts.push(Doc::join(effects, Doc::text(", ")));
        }

        // Where clause
        if !func.where_clause.is_empty() {
            parts.push(Doc::Hardline);
            parts.push(Doc::text("where"));
            parts.push(Doc::Indent(Box::new(Doc::concat(
                func.where_clause
                    .iter()
                    .map(|p| {
                        Doc::concat(vec![
                            Doc::Hardline,
                            self.visit_where_predicate(p),
                            Doc::text(","),
                        ])
                    })
                    .collect(),
            ))));
        }

        // Body
        parts.push(Doc::text(" "));
        parts.push(self.visit_block(&func.body));

        Doc::concat(parts)
    }

    /// Visit function parameters
    fn visit_params(&self, params: &[Param]) -> Doc {
        if params.is_empty() {
            return Doc::text("()");
        }

        let param_docs: Vec<Doc> = params.iter().map(|p| self.visit_param(p)).collect();

        if params.len() <= 3 {
            // Single line
            Doc::concat(vec![
                Doc::text("("),
                Doc::join(param_docs, Doc::text(", ")),
                Doc::text(")"),
            ])
        } else {
            // Multi-line
            let param_docs: Vec<Doc> = params
                .iter()
                .map(|p| {
                    Doc::concat(vec![
                        Doc::Hardline,
                        Doc::Indent(Box::new(Doc::concat(vec![
                            self.visit_param(p),
                            Doc::text(","),
                        ]))),
                    ])
                })
                .collect();

            Doc::concat(vec![
                Doc::text("("),
                Doc::concat(param_docs),
                Doc::Hardline,
                Doc::text(")"),
            ])
        }
    }

    /// Visit a single parameter
    fn visit_param(&self, param: &Param) -> Doc {
        let mut parts = Vec::new();

        // Attributes
        for attr in &param.attributes {
            parts.push(Doc::text(format!("#[{}] ", attr.name)));
        }

        if param.is_mut {
            parts.push(Doc::text("mut "));
        }

        parts.push(self.visit_pattern(&param.pattern));
        parts.push(Doc::text(": "));
        parts.push(self.visit_type(&param.ty));

        Doc::concat(parts)
    }

    // ==================== STRUCTS ====================

    /// Visit a struct definition
    pub fn visit_struct(&self, s: &StructDef) -> Doc {
        let mut parts = Vec::new();

        // Attributes
        for attr in &s.attributes {
            parts.push(Doc::text(format!("#[{}]", attr.name)));
            parts.push(Doc::Hardline);
        }

        // Visibility
        if matches!(s.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        // Linearity
        if s.modifiers.linear {
            parts.push(Doc::text("linear "));
        } else if s.modifiers.affine {
            parts.push(Doc::text("affine "));
        }

        parts.push(Doc::text(format!("struct {}", s.name)));

        // Generics
        if !s.generics.params.is_empty() {
            parts.push(self.visit_generics(&s.generics.params));
        }

        // Where clause
        if !s.where_clause.is_empty() {
            parts.push(Doc::Hardline);
            parts.push(Doc::text("where"));
            parts.push(Doc::Indent(Box::new(Doc::concat(
                s.where_clause
                    .iter()
                    .map(|p| {
                        Doc::concat(vec![
                            Doc::Hardline,
                            self.visit_where_predicate(p),
                            Doc::text(","),
                        ])
                    })
                    .collect(),
            ))));
        }

        // Fields
        if s.fields.is_empty() {
            parts.push(Doc::text(" {}"));
        } else {
            parts.push(Doc::text(" {"));
            for field in &s.fields {
                parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                    Doc::Hardline,
                    self.visit_field(field),
                    Doc::text(","),
                ]))));
            }
            parts.push(Doc::Hardline);
            parts.push(Doc::text("}"));
        }

        Doc::concat(parts)
    }

    /// Visit a struct field
    fn visit_field(&self, field: &crate::ast::FieldDef) -> Doc {
        let mut parts = Vec::new();

        // Attributes
        for attr in &field.attributes {
            parts.push(Doc::text(format!("#[{}] ", attr.name)));
        }

        // Visibility
        if matches!(field.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("{}: ", field.name)));
        parts.push(self.visit_type(&field.ty));

        Doc::concat(parts)
    }

    // ==================== ENUMS ====================

    /// Visit an enum definition
    pub fn visit_enum(&self, e: &EnumDef) -> Doc {
        let mut parts = Vec::new();

        // Visibility
        if matches!(e.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        // Linearity
        if e.modifiers.linear {
            parts.push(Doc::text("linear "));
        } else if e.modifiers.affine {
            parts.push(Doc::text("affine "));
        }

        parts.push(Doc::text(format!("enum {}", e.name)));

        // Generics
        if !e.generics.params.is_empty() {
            parts.push(self.visit_generics(&e.generics.params));
        }

        // Where clause
        if !e.where_clause.is_empty() {
            parts.push(Doc::Hardline);
            parts.push(Doc::text("where"));
            parts.push(Doc::Indent(Box::new(Doc::concat(
                e.where_clause
                    .iter()
                    .map(|p| {
                        Doc::concat(vec![
                            Doc::Hardline,
                            self.visit_where_predicate(p),
                            Doc::text(","),
                        ])
                    })
                    .collect(),
            ))));
        }

        parts.push(Doc::text(" {"));

        for variant in &e.variants {
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                self.visit_variant(variant),
                Doc::text(","),
            ]))));
        }

        parts.push(Doc::Hardline);
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    /// Visit an enum variant
    fn visit_variant(&self, variant: &VariantDef) -> Doc {
        let mut parts = Vec::new();

        parts.push(Doc::text(&variant.name));

        match &variant.data {
            VariantData::Unit => {}
            VariantData::Tuple(types) => {
                parts.push(Doc::text("("));
                let type_docs: Vec<Doc> = types.iter().map(|t| self.visit_type(t)).collect();
                parts.push(Doc::join(type_docs, Doc::text(", ")));
                parts.push(Doc::text(")"));
            }
            VariantData::Struct(fields) => {
                parts.push(Doc::text(" {"));
                for field in fields {
                    parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                        Doc::Hardline,
                        self.visit_field(field),
                        Doc::text(","),
                    ]))));
                }
                parts.push(Doc::Hardline);
                parts.push(Doc::text("}"));
            }
        }

        // GADT return type
        if let Some(gadt) = &variant.gadt_return_type {
            parts.push(Doc::text(": "));
            parts.push(self.visit_type(&gadt.return_type));
        }

        Doc::concat(parts)
    }

    // ==================== TRAITS ====================

    /// Visit a trait definition
    pub fn visit_trait(&self, t: &TraitDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(t.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("trait {}", t.name)));

        if !t.generics.params.is_empty() {
            parts.push(self.visit_generics(&t.generics.params));
        }

        // Supertraits
        if !t.supertraits.is_empty() {
            parts.push(Doc::text(": "));
            let bounds: Vec<Doc> = t
                .supertraits
                .iter()
                .map(|p| Doc::text(p.segments.join("::")))
                .collect();
            parts.push(Doc::join(bounds, Doc::text(" + ")));
        }

        // Where clause
        if !t.where_clause.is_empty() {
            parts.push(Doc::Hardline);
            parts.push(Doc::text("where"));
            parts.push(Doc::Indent(Box::new(Doc::concat(
                t.where_clause
                    .iter()
                    .map(|p| {
                        Doc::concat(vec![
                            Doc::Hardline,
                            self.visit_where_predicate(p),
                            Doc::text(","),
                        ])
                    })
                    .collect(),
            ))));
        }

        parts.push(Doc::text(" {"));

        for item in &t.items {
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                self.visit_trait_item(item),
            ]))));
        }

        if !t.items.is_empty() {
            parts.push(Doc::Hardline);
        }
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    /// Visit a trait item
    fn visit_trait_item(&self, item: &TraitItem) -> Doc {
        match item {
            TraitItem::Fn(f) => {
                let mut parts = Vec::new();
                parts.push(Doc::text(format!("fn {}", f.name)));

                if !f.generics.params.is_empty() {
                    parts.push(self.visit_generics(&f.generics.params));
                }

                // Parameters
                let param_docs: Vec<Doc> = f.params.iter().map(|p| self.visit_param(p)).collect();
                parts.push(Doc::text("("));
                parts.push(Doc::join(param_docs, Doc::text(", ")));
                parts.push(Doc::text(")"));

                if let Some(ret) = &f.return_type {
                    parts.push(Doc::text(" -> "));
                    parts.push(self.visit_type(ret));
                }

                if !f.effects.is_empty() {
                    parts.push(Doc::text(" with "));
                    let effects: Vec<Doc> = f
                        .effects
                        .iter()
                        .map(|e| Doc::text(e.name.segments.join("::")))
                        .collect();
                    parts.push(Doc::join(effects, Doc::text(", ")));
                }

                if let Some(body) = &f.default_body {
                    parts.push(Doc::text(" "));
                    parts.push(self.visit_block(body));
                } else {
                    parts.push(Doc::text(";"));
                }

                Doc::concat(parts)
            }
            TraitItem::Type(t) => {
                let mut parts = Vec::new();
                parts.push(Doc::text(format!("type {}", t.name)));

                if !t.bounds.is_empty() {
                    parts.push(Doc::text(": "));
                    let bounds: Vec<Doc> = t
                        .bounds
                        .iter()
                        .map(|p| Doc::text(p.segments.join("::")))
                        .collect();
                    parts.push(Doc::join(bounds, Doc::text(" + ")));
                }

                if let Some(default) = &t.default {
                    parts.push(Doc::text(" = "));
                    parts.push(self.visit_type(default));
                }

                parts.push(Doc::text(";"));
                Doc::concat(parts)
            }
        }
    }

    // ==================== IMPL ====================

    /// Visit an impl block
    pub fn visit_impl(&self, i: &ImplDef) -> Doc {
        let mut parts = Vec::new();

        parts.push(Doc::text("impl"));

        if !i.generics.params.is_empty() {
            parts.push(self.visit_generics(&i.generics.params));
        }

        if let Some(trait_ref) = &i.trait_ref {
            parts.push(Doc::text(format!(" {} for", trait_ref.segments.join("::"))));
        }

        parts.push(Doc::text(" "));
        parts.push(self.visit_type(&i.target_type));

        // Where clause
        if !i.where_clause.is_empty() {
            parts.push(Doc::Hardline);
            parts.push(Doc::text("where"));
            parts.push(Doc::Indent(Box::new(Doc::concat(
                i.where_clause
                    .iter()
                    .map(|p| {
                        Doc::concat(vec![
                            Doc::Hardline,
                            self.visit_where_predicate(p),
                            Doc::text(","),
                        ])
                    })
                    .collect(),
            ))));
        }

        parts.push(Doc::text(" {"));

        for (idx, item) in i.items.iter().enumerate() {
            if idx > 0 {
                parts.push(Doc::Hardline);
            }
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                self.visit_impl_item(item),
            ]))));
        }

        if !i.items.is_empty() {
            parts.push(Doc::Hardline);
        }
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    /// Visit an impl item
    fn visit_impl_item(&self, item: &ImplItem) -> Doc {
        match item {
            ImplItem::Fn(f) => self.visit_function(f),
            ImplItem::Type(t) => Doc::concat(vec![
                Doc::text(format!("type {} = ", t.name)),
                self.visit_type(&t.ty),
                Doc::text(";"),
            ]),
        }
    }

    // ==================== TYPE ALIAS ====================

    /// Visit a type alias
    pub fn visit_type_alias(&self, t: &TypeAliasDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(t.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("type {}", t.name)));

        if !t.generics.params.is_empty() {
            parts.push(self.visit_generics(&t.generics.params));
        }

        parts.push(Doc::text(" = "));
        parts.push(self.visit_type(&t.ty));
        parts.push(Doc::text(";"));

        Doc::concat(parts)
    }

    // ==================== EFFECTS ====================

    /// Visit an effect definition
    pub fn visit_effect(&self, e: &EffectDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(e.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("effect {}", e.name)));

        if !e.generics.params.is_empty() {
            parts.push(self.visit_generics(&e.generics.params));
        }

        parts.push(Doc::text(" {"));

        for op in &e.operations {
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                Doc::text(format!("fn {}", op.name)),
                Doc::text("("),
                Doc::join(
                    op.params.iter().map(|p| self.visit_param(p)).collect(),
                    Doc::text(", "),
                ),
                Doc::text(")"),
                if let Some(ret) = &op.return_type {
                    Doc::concat(vec![Doc::text(" -> "), self.visit_type(ret)])
                } else {
                    Doc::Empty
                },
                Doc::text(";"),
            ]))));
        }

        if !e.operations.is_empty() {
            parts.push(Doc::Hardline);
        }
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    /// Visit a handler definition
    pub fn visit_handler(&self, h: &HandlerDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(h.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!(
            "handler {} for {}",
            h.name,
            h.effect.segments.join("::")
        )));

        if !h.generics.params.is_empty() {
            parts.push(self.visit_generics(&h.generics.params));
        }

        parts.push(Doc::text(" {"));

        for case in &h.cases {
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                Doc::text(format!("{}", case.name)),
                Doc::text("("),
                Doc::join(
                    case.params.iter().map(|p| self.visit_param(p)).collect(),
                    Doc::text(", "),
                ),
                Doc::text(") => "),
                self.visit_expr(&case.body),
                Doc::text(","),
            ]))));
        }

        if !h.cases.is_empty() {
            parts.push(Doc::Hardline);
        }
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    // ==================== IMPORTS/EXPORTS ====================

    /// Visit an import definition
    pub fn visit_import(&self, u: &ImportDef) -> Doc {
        let mut parts = Vec::new();

        if u.is_reexport {
            parts.push(Doc::text("pub "));
        }
        parts.push(Doc::text("use "));

        let base_path = u.path.segments.join("::");

        match &u.items {
            None => {
                parts.push(Doc::text(format!("{};", base_path)));
            }
            Some(items) if items.len() == 1 && !items[0].is_glob && items[0].alias.is_none() => {
                parts.push(Doc::text(format!("{}::{};", base_path, items[0].name)));
            }
            Some(items) if items.len() == 1 && items[0].is_glob => {
                parts.push(Doc::text(format!("{}::*;", base_path)));
            }
            Some(items) => {
                let item_strs: Vec<String> = items
                    .iter()
                    .map(|item| {
                        if item.is_glob {
                            "*".to_string()
                        } else if let Some(alias) = &item.alias {
                            format!("{} as {}", item.name, alias)
                        } else {
                            item.name.clone()
                        }
                    })
                    .collect();
                if items.len() == 1 {
                    parts.push(Doc::text(format!("{}::{};", base_path, item_strs[0])));
                } else {
                    parts.push(Doc::text(format!(
                        "{}::{{{}}};",
                        base_path,
                        item_strs.join(", ")
                    )));
                }
            }
        }

        Doc::concat(parts)
    }

    /// Visit an export definition
    fn visit_export(&self, e: &crate::ast::ExportDef) -> Doc {
        if e.names.len() == 1 {
            Doc::text(format!("export {};", e.names[0]))
        } else {
            Doc::text(format!("export {{ {} }};", e.names.join(", ")))
        }
    }

    // ==================== EXTERN ====================

    /// Visit an extern block
    pub fn visit_extern(&self, e: &ExternBlock) -> Doc {
        let mut parts = Vec::new();

        parts.push(Doc::text(format!("extern \"{}\" {{", e.abi)));

        for item in &e.items {
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                self.visit_extern_item(item),
            ]))));
        }

        if !e.items.is_empty() {
            parts.push(Doc::Hardline);
        }
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    /// Visit an extern item
    fn visit_extern_item(&self, item: &ExternItem) -> Doc {
        match item {
            ExternItem::Fn(f) => {
                let mut parts = Vec::new();
                parts.push(Doc::text(format!("fn {}", f.name)));
                parts.push(Doc::text("("));

                let param_docs: Vec<Doc> = f.params.iter().map(|p| self.visit_param(p)).collect();
                parts.push(Doc::join(param_docs, Doc::text(", ")));

                if f.is_variadic {
                    if !f.params.is_empty() {
                        parts.push(Doc::text(", "));
                    }
                    parts.push(Doc::text("..."));
                }

                parts.push(Doc::text(")"));

                if let Some(ret) = &f.return_type {
                    parts.push(Doc::text(" -> "));
                    parts.push(self.visit_type(ret));
                }

                parts.push(Doc::text(";"));
                Doc::concat(parts)
            }
            ExternItem::Static(s) => {
                let mut parts = Vec::new();
                if s.is_mut {
                    parts.push(Doc::text("static mut "));
                } else {
                    parts.push(Doc::text("static "));
                }
                parts.push(Doc::text(format!("{}: ", s.name)));
                parts.push(self.visit_type(&s.ty));
                parts.push(Doc::text(";"));
                Doc::concat(parts)
            }
            ExternItem::Type(t) => Doc::text(format!("type {};", t.name)),
        }
    }

    // ==================== GLOBAL ====================

    /// Visit a global definition
    pub fn visit_global(&self, g: &GlobalDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(g.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        if g.is_const {
            parts.push(Doc::text("const "));
        } else if g.is_mut {
            parts.push(Doc::text("static mut "));
        } else {
            parts.push(Doc::text("static "));
        }

        parts.push(self.visit_pattern(&g.pattern));

        if let Some(ty) = &g.ty {
            parts.push(Doc::text(": "));
            parts.push(self.visit_type(ty));
        }

        parts.push(Doc::text(" = "));
        parts.push(self.visit_expr(&g.value));
        parts.push(Doc::text(";"));

        Doc::concat(parts)
    }

    // ==================== MODULE ====================

    /// Visit a module definition
    fn visit_module(&self, m: &ModuleDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(m.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        if let Some(items) = &m.items {
            // Inline module
            parts.push(Doc::text(format!("module {} {{", m.name)));

            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    parts.push(Doc::Hardline);
                }
                parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                    Doc::Hardline,
                    self.visit_item(item),
                ]))));
            }

            if !items.is_empty() {
                parts.push(Doc::Hardline);
            }
            parts.push(Doc::text("}"));
        } else {
            // File-based module
            parts.push(Doc::text(format!("mod {};", m.name)));
        }

        Doc::concat(parts)
    }

    // ==================== SCIENTIFIC ITEMS ====================

    /// Visit an ontology import
    fn visit_ontology_import(&self, o: &crate::ast::OntologyImportDef) -> Doc {
        let mut parts = Vec::new();
        parts.push(Doc::text(format!(
            "ontology {} from \"{}\"",
            o.prefix, o.source
        )));
        if let Some(alias) = &o.alias {
            parts.push(Doc::text(format!(" as {}", alias)));
        }
        parts.push(Doc::text(";"));
        Doc::concat(parts)
    }

    /// Visit an alignment declaration
    fn visit_align(&self, a: &crate::ast::AlignDef) -> Doc {
        Doc::text(format!(
            "align {}:{} ~ {}:{} with distance {};",
            a.type1.prefix, a.type1.term, a.type2.prefix, a.type2.term, a.distance
        ))
    }

    /// Visit an ODE definition
    fn visit_ode(&self, o: &OdeDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(o.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("ode {} {{ ... }}", o.name)));

        Doc::concat(parts)
    }

    /// Visit a PDE definition
    fn visit_pde(&self, p: &PdeDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(p.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("pde {} {{ ... }}", p.name)));

        Doc::concat(parts)
    }

    /// Visit a causal model
    fn visit_causal_model(&self, c: &CausalModelDef) -> Doc {
        let mut parts = Vec::new();

        if matches!(c.visibility, Visibility::Public) {
            parts.push(Doc::text("pub "));
        }

        parts.push(Doc::text(format!("causal model {} {{ ... }}", c.name)));

        Doc::concat(parts)
    }

    // ==================== GENERICS ====================

    /// Visit generic parameters
    fn visit_generics(&self, params: &[GenericParam]) -> Doc {
        let param_docs: Vec<Doc> = params.iter().map(|p| self.visit_generic_param(p)).collect();
        Doc::concat(vec![
            Doc::text("<"),
            Doc::join(param_docs, Doc::text(", ")),
            Doc::text(">"),
        ])
    }

    /// Visit a single generic parameter
    fn visit_generic_param(&self, param: &GenericParam) -> Doc {
        match param {
            GenericParam::Type {
                name,
                bounds,
                default,
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text(name));

                if !bounds.is_empty() {
                    parts.push(Doc::text(": "));
                    let bounds_str: Vec<Doc> = bounds
                        .iter()
                        .map(|p| Doc::text(p.segments.join("::")))
                        .collect();
                    parts.push(Doc::join(bounds_str, Doc::text(" + ")));
                }

                if let Some(def) = default {
                    parts.push(Doc::text(" = "));
                    parts.push(self.visit_type(def));
                }

                Doc::concat(parts)
            }
            GenericParam::Const { name, ty } => Doc::concat(vec![
                Doc::text(format!("const {}: ", name)),
                self.visit_type(ty),
            ]),
            GenericParam::Effect { name } => Doc::text(format!("effect {}", name)),
        }
    }

    /// Visit a where predicate
    fn visit_where_predicate(&self, pred: &WherePredicate) -> Doc {
        let mut parts = Vec::new();
        parts.push(self.visit_type(&pred.ty));
        parts.push(Doc::text(": "));

        let bounds: Vec<Doc> = pred
            .bounds
            .iter()
            .map(|p| Doc::text(p.segments.join("::")))
            .collect();
        parts.push(Doc::join(bounds, Doc::text(" + ")));

        Doc::concat(parts)
    }

    // ==================== TYPES ====================

    /// Visit a type expression
    pub fn visit_type(&self, ty: &TypeExpr) -> Doc {
        match ty {
            TypeExpr::Unit => Doc::text("()"),
            TypeExpr::Never => Doc::text("!"),
            TypeExpr::SelfType => Doc::text("Self"),
            TypeExpr::Named { path, args, unit } => {
                let mut parts = Vec::new();
                parts.push(Doc::text(path.segments.join("::")));

                if !args.is_empty() {
                    parts.push(Doc::text("<"));
                    let args_docs: Vec<Doc> = args.iter().map(|a| self.visit_type(a)).collect();
                    parts.push(Doc::join(args_docs, Doc::text(", ")));
                    parts.push(Doc::text(">"));
                }

                if let Some(u) = unit {
                    parts.push(Doc::text(format!("<{}>", u)));
                }

                Doc::concat(parts)
            }
            TypeExpr::Reference { mutable, inner } => {
                // Sounio uses &! for mutable references
                if *mutable {
                    Doc::concat(vec![Doc::text("&!"), self.visit_type(inner)])
                } else {
                    Doc::concat(vec![Doc::text("&"), self.visit_type(inner)])
                }
            }
            TypeExpr::RawPointer { mutable, inner } => {
                if *mutable {
                    Doc::concat(vec![Doc::text("*mut "), self.visit_type(inner)])
                } else {
                    Doc::concat(vec![Doc::text("*const "), self.visit_type(inner)])
                }
            }
            TypeExpr::Array { element, size } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("["));
                parts.push(self.visit_type(element));
                if let Some(s) = size {
                    parts.push(Doc::text("; "));
                    parts.push(self.visit_expr(s));
                }
                parts.push(Doc::text("]"));
                Doc::concat(parts)
            }
            TypeExpr::Tuple(elements) => {
                let elems: Vec<Doc> = elements.iter().map(|e| self.visit_type(e)).collect();
                Doc::concat(vec![
                    Doc::text("("),
                    Doc::join(elems, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            TypeExpr::Function {
                abi,
                params,
                return_type,
                effects,
            } => {
                let mut parts = Vec::new();
                // Add ABI prefix for extern function pointers
                if let Some(a) = abi {
                    parts.push(Doc::text(format!("extern \"{}\" ", a)));
                }
                parts.push(Doc::text("fn("));

                let ps: Vec<Doc> = params.iter().map(|p| self.visit_type(p)).collect();
                parts.push(Doc::join(ps, Doc::text(", ")));
                parts.push(Doc::text(") -> "));
                parts.push(self.visit_type(return_type));

                if !effects.is_empty() {
                    parts.push(Doc::text(" with "));
                    let effs: Vec<Doc> = effects
                        .iter()
                        .map(|e| Doc::text(e.name.segments.join("::")))
                        .collect();
                    parts.push(Doc::join(effs, Doc::text(", ")));
                }

                Doc::concat(parts)
            }
            TypeExpr::Infer => Doc::text("_"),
            TypeExpr::Knowledge { value_type, .. } => Doc::concat(vec![
                Doc::text("Knowledge["),
                self.visit_type(value_type),
                Doc::text("]"),
            ]),
            TypeExpr::Quantity { numeric_type, unit } => {
                let unit_str = self.format_unit(unit);
                Doc::concat(vec![
                    Doc::text("Quantity["),
                    self.visit_type(numeric_type),
                    Doc::text(format!(", {}]", unit_str)),
                ])
            }
            TypeExpr::Tensor {
                element_type,
                shape,
            } => {
                let shape_str = shape
                    .iter()
                    .map(|d| match d {
                        crate::ast::TensorDim::Named(n) => n.clone(),
                        crate::ast::TensorDim::Fixed(s) => s.to_string(),
                        crate::ast::TensorDim::Dynamic => "_".to_string(),
                        crate::ast::TensorDim::Expr(_) => "<expr>".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                Doc::concat(vec![
                    Doc::text("Tensor["),
                    self.visit_type(element_type),
                    Doc::text(format!(", ({})]", shape_str)),
                ])
            }
            TypeExpr::Tile {
                element_type,
                tile_m,
                tile_n,
                layout,
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("tile<"));
                parts.push(self.visit_type(element_type));
                parts.push(Doc::text(format!(", {}, {}", tile_m, tile_n)));
                if let Some(l) = layout {
                    parts.push(Doc::text(format!(", \"{}\"", l)));
                }
                parts.push(Doc::text(">"));
                Doc::concat(parts)
            }
            TypeExpr::Ontology { ontology, term } => {
                if let Some(t) = term {
                    Doc::text(format!("OntologyTerm[{}:{}]", ontology, t))
                } else {
                    Doc::text(format!("OntologyTerm[{}]", ontology))
                }
            }
            TypeExpr::Linear { inner, linearity } => {
                let kind = match linearity {
                    crate::ast::LinearityKind::Linear => "linear",
                    crate::ast::LinearityKind::Affine => "affine",
                    crate::ast::LinearityKind::Relevant => "relevant",
                    crate::ast::LinearityKind::Unrestricted => "",
                };
                if kind.is_empty() {
                    self.visit_type(inner)
                } else {
                    Doc::concat(vec![
                        self.visit_type(inner),
                        Doc::text(format!(" @ {}", kind)),
                    ])
                }
            }
            TypeExpr::Effected { inner, effects } => {
                let effects_str = effects.effects.join(", ");
                Doc::concat(vec![
                    self.visit_type(inner),
                    Doc::text(format!(" ! {{{}}}", effects_str)),
                ])
            }
            TypeExpr::Refinement {
                var,
                base_type,
                predicate,
            } => Doc::concat(vec![
                Doc::text(format!("{{ {}: ", var)),
                self.visit_type(base_type),
                Doc::text(" | "),
                self.visit_expr(predicate),
                Doc::text(" }"),
            ]),
            TypeExpr::Forall { vars, inner } => {
                let var_strs: Vec<String> = vars
                    .iter()
                    .map(|v| {
                        if v.bounds.is_empty() {
                            v.name.clone()
                        } else {
                            format!(
                                "{}: {}",
                                v.name,
                                v.bounds
                                    .iter()
                                    .map(|b| b.to_string())
                                    .collect::<Vec<_>>()
                                    .join(" + ")
                            )
                        }
                    })
                    .collect();
                Doc::concat(vec![
                    Doc::text("forall "),
                    Doc::text(var_strs.join(", ")),
                    Doc::text(". "),
                    self.visit_type(inner),
                ])
            }
            TypeExpr::ScientificArray { element_type, dim } => Doc::concat(vec![
                Doc::text("Array<"),
                self.visit_type(element_type),
                Doc::text(", "),
                Doc::text(self.format_tensor_dim(dim)),
                Doc::text(">"),
            ]),
            TypeExpr::ScientificMatrix {
                element_type,
                rows,
                cols,
            } => Doc::concat(vec![
                Doc::text("Matrix<"),
                self.visit_type(element_type),
                Doc::text(", "),
                Doc::text(self.format_tensor_dim(rows)),
                Doc::text(", "),
                Doc::text(self.format_tensor_dim(cols)),
                Doc::text(">"),
            ]),
        }
    }

    /// Format a unit expression
    fn format_unit(&self, unit: &crate::ast::UnitExpr) -> String {
        if unit.base_units.is_empty() {
            return "1".to_string();
        }

        unit.base_units
            .iter()
            .map(|(name, exp)| {
                if *exp == 1 {
                    name.clone()
                } else {
                    format!("{}^{}", name, exp)
                }
            })
            .collect::<Vec<_>>()
            .join("*")
    }

    /// Format a tensor dimension
    fn format_tensor_dim(&self, dim: &crate::ast::TensorDim) -> String {
        use crate::ast::TensorDim;
        match dim {
            TensorDim::Named(name) => name.clone(),
            TensorDim::Fixed(size) => size.to_string(),
            TensorDim::Dynamic => "_".to_string(),
            TensorDim::Expr(expr) => {
                // For expressions, format them - this is a simplification
                format!("{:?}", expr)
            }
        }
    }

    // ==================== BLOCKS ====================

    /// Visit a block
    pub fn visit_block(&self, block: &Block) -> Doc {
        if block.stmts.is_empty() {
            return Doc::text("{}");
        }

        let mut parts = Vec::new();
        parts.push(Doc::text("{"));

        for stmt in &block.stmts {
            parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                Doc::Hardline,
                self.visit_stmt(stmt),
            ]))));
        }

        parts.push(Doc::Hardline);
        parts.push(Doc::text("}"));

        Doc::concat(parts)
    }

    // ==================== STATEMENTS ====================

    /// Visit a statement
    pub fn visit_stmt(&self, stmt: &Stmt) -> Doc {
        match stmt {
            Stmt::Let {
                is_mut,
                pattern,
                ty,
                value,
            } => {
                let mut parts = Vec::new();

                // Sounio uses `var` for mutable, `let` for immutable
                if *is_mut {
                    parts.push(Doc::text("var "));
                } else {
                    parts.push(Doc::text("let "));
                }

                parts.push(self.visit_pattern(pattern));

                if let Some(t) = ty {
                    parts.push(Doc::text(": "));
                    parts.push(self.visit_type(t));
                }

                if let Some(v) = value {
                    parts.push(Doc::text(" = "));
                    parts.push(self.visit_expr(v));
                }

                parts.push(Doc::text(";"));
                Doc::concat(parts)
            }
            Stmt::Expr { expr, has_semi } => {
                if *has_semi {
                    Doc::concat(vec![self.visit_expr(expr), Doc::text(";")])
                } else {
                    self.visit_expr(expr)
                }
            }
            Stmt::Assign { target, op, value } => Doc::concat(vec![
                self.visit_expr(target),
                Doc::text(format!(" {} ", self.format_assign_op(op))),
                self.visit_expr(value),
                Doc::text(";"),
            ]),
            Stmt::Empty => Doc::Empty,
            Stmt::MacroInvocation(m) => Doc::text(format!("{}!(...);", m.name)),
            Stmt::LocalExtern(ext) => {
                // Format local extern block
                let abi_str = match &ext.abi {
                    crate::ast::Abi::C => "\"C\"",
                    crate::ast::Abi::System => "\"system\"",
                    crate::ast::Abi::Rust => "\"Rust\"",
                    crate::ast::Abi::PlatformIntrinsic => "\"platform-intrinsic\"",
                    _ => "\"C\"", // default to C for unknown ABIs
                };
                Doc::text(format!("extern {} {{ ... }}", abi_str))
            }
        }
    }

    /// Format an assignment operator
    fn format_assign_op(&self, op: &AssignOp) -> &'static str {
        match op {
            AssignOp::Assign => "=",
            AssignOp::AddAssign => "+=",
            AssignOp::SubAssign => "-=",
            AssignOp::MulAssign => "*=",
            AssignOp::DivAssign => "/=",
            AssignOp::RemAssign => "%=",
            AssignOp::BitAndAssign => "&=",
            AssignOp::BitOrAssign => "|=",
            AssignOp::BitXorAssign => "^=",
            AssignOp::ShlAssign => "<<=",
            AssignOp::ShrAssign => ">>=",
        }
    }

    // ==================== PATTERNS ====================

    /// Visit a pattern
    pub fn visit_pattern(&self, pattern: &Pattern) -> Doc {
        match pattern {
            Pattern::Wildcard => Doc::text("_"),
            Pattern::Literal(lit) => self.visit_literal(lit),
            Pattern::Binding { name, mutable } => {
                if *mutable {
                    Doc::text(format!("mut {}", name))
                } else {
                    Doc::text(name)
                }
            }
            Pattern::Tuple(patterns) => {
                let ps: Vec<Doc> = patterns.iter().map(|p| self.visit_pattern(p)).collect();
                Doc::concat(vec![
                    Doc::text("("),
                    Doc::join(ps, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Pattern::Struct { path, fields } => {
                let fs: Vec<Doc> = fields
                    .iter()
                    .map(|(name, pat)| {
                        Doc::concat(vec![
                            Doc::text(format!("{}: ", name)),
                            self.visit_pattern(pat),
                        ])
                    })
                    .collect();
                Doc::concat(vec![
                    Doc::text(path.segments.join("::")),
                    Doc::text(" { "),
                    Doc::join(fs, Doc::text(", ")),
                    Doc::text(" }"),
                ])
            }
            Pattern::Enum { path, patterns } => {
                let mut parts = Vec::new();
                parts.push(Doc::text(path.segments.join("::")));

                if let Some(ps) = patterns {
                    parts.push(Doc::text("("));
                    let pats: Vec<Doc> = ps.iter().map(|p| self.visit_pattern(p)).collect();
                    parts.push(Doc::join(pats, Doc::text(", ")));
                    parts.push(Doc::text(")"));
                }

                Doc::concat(parts)
            }
            Pattern::Or(patterns) => {
                let ps: Vec<Doc> = patterns.iter().map(|p| self.visit_pattern(p)).collect();
                Doc::join(ps, Doc::text(" | "))
            }
        }
    }

    // ==================== EXPRESSIONS ====================

    /// Visit an expression
    pub fn visit_expr(&self, expr: &Expr) -> Doc {
        match expr {
            Expr::Literal { value, .. } => self.visit_literal(value),
            Expr::Path { path, .. } => Doc::text(path.segments.join("::")),
            Expr::Binary {
                op, left, right, ..
            } => Doc::Group(Box::new(Doc::concat(vec![
                self.visit_expr(left),
                Doc::text(format!(" {} ", self.format_binop(op))),
                self.visit_expr(right),
            ]))),
            Expr::Unary { op, expr, .. } => {
                Doc::concat(vec![Doc::text(self.format_unop(op)), self.visit_expr(expr)])
            }
            Expr::Call { callee, args, .. } => {
                let args_docs: Vec<Doc> = args.iter().map(|a| self.visit_expr(a)).collect();
                Doc::concat(vec![
                    self.visit_expr(callee),
                    Doc::text("("),
                    Doc::join(args_docs, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => {
                let args_docs: Vec<Doc> = args.iter().map(|a| self.visit_expr(a)).collect();
                Doc::concat(vec![
                    self.visit_expr(receiver),
                    Doc::text(format!(".{}(", method)),
                    Doc::join(args_docs, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Expr::Field { base, field, .. } => Doc::concat(vec![
                self.visit_expr(base),
                Doc::text(format!(".{}", field)),
            ]),
            Expr::TupleField { base, index, .. } => Doc::concat(vec![
                self.visit_expr(base),
                Doc::text(format!(".{}", index)),
            ]),
            Expr::Index { base, index, .. } => Doc::concat(vec![
                self.visit_expr(base),
                Doc::text("["),
                self.visit_expr(index),
                Doc::text("]"),
            ]),
            Expr::Cast { expr, ty, .. } => Doc::concat(vec![
                self.visit_expr(expr),
                Doc::text(" as "),
                self.visit_type(ty),
            ]),
            Expr::Block { block, .. } => self.visit_block(block),
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("if "));
                parts.push(self.visit_expr(condition));
                parts.push(Doc::text(" "));
                parts.push(self.visit_block(then_branch));

                if let Some(else_b) = else_branch {
                    parts.push(Doc::text(" else "));
                    parts.push(self.visit_expr(else_b));
                }

                Doc::concat(parts)
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("match "));
                parts.push(self.visit_expr(scrutinee));
                parts.push(Doc::text(" {"));

                for arm in arms {
                    parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                        Doc::Hardline,
                        self.visit_match_arm(arm),
                    ]))));
                }

                parts.push(Doc::Hardline);
                parts.push(Doc::text("}"));
                Doc::concat(parts)
            }
            Expr::Loop { body, .. } => {
                Doc::concat(vec![Doc::text("loop "), self.visit_block(body)])
            }
            Expr::While {
                condition, body, ..
            } => Doc::concat(vec![
                Doc::text("while "),
                self.visit_expr(condition),
                Doc::text(" "),
                self.visit_block(body),
            ]),
            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => Doc::concat(vec![
                Doc::text("for "),
                self.visit_pattern(pattern),
                Doc::text(" in "),
                self.visit_expr(iter),
                Doc::text(" "),
                self.visit_block(body),
            ]),
            Expr::Return { value, .. } => {
                if let Some(v) = value {
                    Doc::concat(vec![Doc::text("return "), self.visit_expr(v)])
                } else {
                    Doc::text("return")
                }
            }
            Expr::Break { value, .. } => {
                if let Some(v) = value {
                    Doc::concat(vec![Doc::text("break "), self.visit_expr(v)])
                } else {
                    Doc::text("break")
                }
            }
            Expr::Continue { .. } => Doc::text("continue"),
            Expr::Closure {
                params,
                return_type,
                body,
                ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("|"));

                let ps: Vec<Doc> = params
                    .iter()
                    .map(|(name, ty)| {
                        if let Some(t) = ty {
                            Doc::concat(vec![Doc::text(format!("{}: ", name)), self.visit_type(t)])
                        } else {
                            Doc::text(name)
                        }
                    })
                    .collect();
                parts.push(Doc::join(ps, Doc::text(", ")));
                parts.push(Doc::text("|"));

                if let Some(ret) = return_type {
                    parts.push(Doc::text(" -> "));
                    parts.push(self.visit_type(ret));
                }

                parts.push(Doc::text(" "));
                parts.push(self.visit_expr(body));

                Doc::concat(parts)
            }
            Expr::Tuple { elements, .. } => {
                let elems: Vec<Doc> = elements.iter().map(|e| self.visit_expr(e)).collect();
                Doc::concat(vec![
                    Doc::text("("),
                    Doc::join(elems, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Expr::Array { elements, .. } => {
                if elements.is_empty() {
                    return Doc::text("[]");
                }

                let elems: Vec<Doc> = elements.iter().map(|e| self.visit_expr(e)).collect();
                Doc::Group(Box::new(Doc::concat(vec![
                    Doc::text("["),
                    Doc::join(elems, Doc::text(", ")),
                    Doc::text("]"),
                ])))
            }
            Expr::Range {
                start,
                end,
                inclusive,
                ..
            } => {
                let mut parts = Vec::new();
                if let Some(s) = start {
                    parts.push(self.visit_expr(s));
                }
                parts.push(Doc::text(if *inclusive { "..=" } else { ".." }));
                if let Some(e) = end {
                    parts.push(self.visit_expr(e));
                }
                Doc::concat(parts)
            }
            Expr::StructLit { path, fields, .. } => {
                let fs: Vec<Doc> = fields
                    .iter()
                    .map(|(name, expr)| {
                        Doc::concat(vec![
                            Doc::text(format!("{}: ", name)),
                            self.visit_expr(expr),
                        ])
                    })
                    .collect();

                if fields.is_empty() {
                    Doc::concat(vec![Doc::text(path.segments.join("::")), Doc::text(" {}")])
                } else {
                    Doc::concat(vec![
                        Doc::text(path.segments.join("::")),
                        Doc::text(" { "),
                        Doc::join(fs, Doc::text(", ")),
                        Doc::text(" }"),
                    ])
                }
            }
            Expr::Try { expr, .. } => Doc::concat(vec![self.visit_expr(expr), Doc::text("?")]),
            Expr::Perform {
                effect, op, args, ..
            } => {
                let args_docs: Vec<Doc> = args.iter().map(|a| self.visit_expr(a)).collect();
                Doc::concat(vec![
                    Doc::text(format!("perform {}::{}", effect.segments.join("::"), op)),
                    Doc::text("("),
                    Doc::join(args_docs, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Expr::Handle { expr, handler, .. } => Doc::concat(vec![
                Doc::text("handle "),
                self.visit_expr(expr),
                Doc::text(format!(" with {}", handler.segments.join("::"))),
            ]),
            Expr::Resume { value, .. } => Doc::concat(vec![
                Doc::text("resume("),
                self.visit_expr(value),
                Doc::text(")"),
            ]),
            Expr::Sample { distribution, .. } => Doc::concat(vec![
                Doc::text("sample("),
                self.visit_expr(distribution),
                Doc::text(")"),
            ]),
            Expr::Await { expr, .. } => {
                Doc::concat(vec![self.visit_expr(expr), Doc::text(".await")])
            }
            Expr::AsyncBlock { block, .. } => {
                Doc::concat(vec![Doc::text("async "), self.visit_block(block)])
            }
            Expr::UnsafeBlock { block, .. } => {
                Doc::concat(vec![Doc::text("unsafe "), self.visit_block(block)])
            }
            Expr::AsyncClosure {
                params,
                return_type,
                body,
                ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("async |"));

                let ps: Vec<Doc> = params
                    .iter()
                    .map(|(name, ty)| {
                        if let Some(t) = ty {
                            Doc::concat(vec![Doc::text(format!("{}: ", name)), self.visit_type(t)])
                        } else {
                            Doc::text(name)
                        }
                    })
                    .collect();
                parts.push(Doc::join(ps, Doc::text(", ")));
                parts.push(Doc::text("|"));

                if let Some(ret) = return_type {
                    parts.push(Doc::text(" -> "));
                    parts.push(self.visit_type(ret));
                }

                parts.push(Doc::text(" "));
                parts.push(self.visit_expr(body));

                Doc::concat(parts)
            }
            Expr::Spawn { expr, .. } => {
                Doc::concat(vec![Doc::text("spawn "), self.visit_expr(expr)])
            }
            Expr::Select { arms, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("select {"));

                for arm in arms {
                    parts.push(Doc::Indent(Box::new(Doc::concat(vec![
                        Doc::Hardline,
                        self.visit_select_arm(arm),
                    ]))));
                }

                parts.push(Doc::Hardline);
                parts.push(Doc::text("}"));
                Doc::concat(parts)
            }
            Expr::Join { futures, .. } => {
                let fs: Vec<Doc> = futures.iter().map(|f| self.visit_expr(f)).collect();
                Doc::concat(vec![
                    Doc::text("join!("),
                    Doc::join(fs, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Expr::MacroInvocation(m) => Doc::text(format!("{}!(...)", m.name)),
            Expr::OntologyTerm { ontology, term, .. } => {
                Doc::text(format!("{}:{}", ontology, term))
            }
            Expr::Do { interventions, .. } => {
                let ivs: Vec<Doc> = interventions
                    .iter()
                    .map(|(var, val)| {
                        Doc::concat(vec![Doc::text(format!("{} = ", var)), self.visit_expr(val)])
                    })
                    .collect();
                Doc::concat(vec![
                    Doc::text("do("),
                    Doc::join(ivs, Doc::text(", ")),
                    Doc::text(")"),
                ])
            }
            Expr::Counterfactual {
                factual,
                intervention,
                outcome,
                ..
            } => Doc::concat(vec![Doc::text("counterfactual { ... }")]),
            Expr::KnowledgeExpr { value, epsilon, .. } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("Knowledge::new("));
                parts.push(self.visit_expr(value));

                if let Some(eps) = epsilon {
                    parts.push(Doc::text(", "));
                    parts.push(self.visit_expr(eps));
                }

                parts.push(Doc::text(")"));
                Doc::concat(parts)
            }
            Expr::Uncertain {
                value, uncertainty, ..
            } => Doc::concat(vec![
                self.visit_expr(value),
                Doc::text(" +- "),
                self.visit_expr(uncertainty),
            ]),
            Expr::GpuAnnotated {
                expr, annotation, ..
            } => {
                let kind = match annotation.kind {
                    crate::ast::GpuAnnotationKind::Gpu => "gpu",
                    crate::ast::GpuAnnotationKind::GpuEpistemic => "gpu.epistemic",
                    crate::ast::GpuAnnotationKind::GpuReduction => "gpu.reduction",
                    crate::ast::GpuAnnotationKind::GpuParallel => "gpu.parallel",
                    crate::ast::GpuAnnotationKind::GpuShared => "gpu.shared",
                    crate::ast::GpuAnnotationKind::GpuDevice => "gpu.device",
                };
                Doc::concat(vec![
                    self.visit_expr(expr),
                    Doc::text(format!(" @ {}", kind)),
                ])
            }
            Expr::Observe {
                data, distribution, ..
            } => Doc::concat(vec![
                Doc::text("observe("),
                self.visit_expr(data),
                Doc::text(" ~ "),
                self.visit_expr(distribution),
                Doc::text(")"),
            ]),
            Expr::Query {
                target,
                given,
                interventions,
                ..
            } => {
                let mut parts = Vec::new();
                parts.push(Doc::text("P("));
                parts.push(self.visit_expr(target));

                if !given.is_empty() || !interventions.is_empty() {
                    parts.push(Doc::text(" | "));

                    let givens: Vec<Doc> = given.iter().map(|g| self.visit_expr(g)).collect();
                    if !givens.is_empty() {
                        parts.push(Doc::join(givens, Doc::text(", ")));
                    }

                    if !interventions.is_empty() {
                        if !given.is_empty() {
                            parts.push(Doc::text(", "));
                        }
                        let ivs: Vec<Doc> = interventions
                            .iter()
                            .map(|(var, val)| {
                                Doc::concat(vec![
                                    Doc::text(format!("do({} = ", var)),
                                    self.visit_expr(val),
                                    Doc::text(")"),
                                ])
                            })
                            .collect();
                        parts.push(Doc::join(ivs, Doc::text(", ")));
                    }
                }

                parts.push(Doc::text(")"));
                Doc::concat(parts)
            }
        }
    }

    /// Visit a match arm
    fn visit_match_arm(&self, arm: &MatchArm) -> Doc {
        let mut parts = Vec::new();
        parts.push(self.visit_pattern(&arm.pattern));

        if let Some(guard) = &arm.guard {
            parts.push(Doc::text(" if "));
            parts.push(self.visit_expr(guard));
        }

        parts.push(Doc::text(" => "));
        parts.push(self.visit_expr(&arm.body));
        parts.push(Doc::text(","));

        Doc::concat(parts)
    }

    /// Visit a select arm
    fn visit_select_arm(&self, arm: &SelectArm) -> Doc {
        let mut parts = Vec::new();
        parts.push(self.visit_expr(&arm.future));
        parts.push(Doc::text(" => "));
        parts.push(self.visit_pattern(&arm.pattern));

        if let Some(guard) = &arm.guard {
            parts.push(Doc::text(" if "));
            parts.push(self.visit_expr(guard));
        }

        parts.push(Doc::text(" => "));
        parts.push(self.visit_expr(&arm.body));
        parts.push(Doc::text(","));

        Doc::concat(parts)
    }

    /// Visit a literal
    fn visit_literal(&self, lit: &Literal) -> Doc {
        match lit {
            Literal::Unit => Doc::text("()"),
            Literal::Bool(b) => Doc::text(if *b { "true" } else { "false" }),
            Literal::Int(i) => Doc::text(i.to_string()),
            Literal::Float(f) => {
                let s = f.to_string();
                if s.contains('.') || s.contains('e') || s.contains('E') {
                    Doc::text(s)
                } else {
                    Doc::text(format!("{}.0", s))
                }
            }
            Literal::Char(c) => Doc::text(format!("'{}'", c.escape_default())),
            Literal::String(s) => Doc::text(format!("\"{}\"", s.escape_default())),
            Literal::CString(s) => Doc::text(format!("c\"{}\"", s.escape_default())),
            Literal::IntUnit(i, u) => Doc::text(format!("{}_{}", i, u)),
            Literal::FloatUnit(f, u) => Doc::text(format!("{}_{}", f, u)),
            Literal::TypedInt(i, suffix) => Doc::text(format!("{}{}", i, suffix)),
            Literal::TypedFloat(f, suffix) => {
                let s = f.to_string();
                let num = if s.contains('.') || s.contains('e') || s.contains('E') {
                    s
                } else {
                    format!("{}.0", s)
                };
                Doc::text(format!("{}{}", num, suffix))
            }
        }
    }

    /// Format a binary operator
    fn format_binop(&self, op: &BinaryOp) -> &'static str {
        match op {
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Rem => "%",
            BinaryOp::And => "&&",
            BinaryOp::Or => "||",
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::Shl => "<<",
            BinaryOp::Shr => ">>",
            BinaryOp::Eq => "==",
            BinaryOp::Ne => "!=",
            BinaryOp::Lt => "<",
            BinaryOp::Le => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::Ge => ">=",
            BinaryOp::PlusMinus => "+-",
            BinaryOp::Concat => "++",
        }
    }

    /// Format a unary operator
    fn format_unop(&self, op: &UnaryOp) -> &'static str {
        match op {
            UnaryOp::Neg => "-",
            UnaryOp::Not => "!",
            UnaryOp::Ref => "&",
            UnaryOp::RefMut => "&!",
            UnaryOp::Deref => "*",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Block, FnModifiers, Generics, Path, Visibility};
    use crate::common::NodeId;

    fn default_config() -> FormatConfig {
        FormatConfig::default()
    }

    #[test]
    fn test_visit_simple_function() {
        let config = default_config();
        let visitor = FormatVisitor::new(&config);

        let func = FnDef {
            id: NodeId::dummy(),
            visibility: Visibility::Public,
            modifiers: FnModifiers::default(),
            attributes: vec![],
            name: "foo".to_string(),
            generics: Generics::default(),
            params: vec![],
            return_type: None,
            effects: vec![],
            where_clause: vec![],
            requires: vec![],
            ensures: vec![],
            body: Block { stmts: vec![] },
            doc: None,
            span: crate::common::Span::dummy(),
        };

        let doc = visitor.visit_function(&func);
        assert!(!doc.is_empty());
    }

    #[test]
    fn test_visit_literal() {
        let config = default_config();
        let visitor = FormatVisitor::new(&config);

        // Test various literals
        let bool_doc = visitor.visit_literal(&Literal::Bool(true));
        let int_doc = visitor.visit_literal(&Literal::Int(42));
        let str_doc = visitor.visit_literal(&Literal::String("hello".to_string()));
        let unit_doc = visitor.visit_literal(&Literal::IntUnit(500, "mg".to_string()));

        // Just verify they don't panic and produce non-empty docs
        assert!(!bool_doc.is_empty());
        assert!(!int_doc.is_empty());
        assert!(!str_doc.is_empty());
        assert!(!unit_doc.is_empty());
    }

    #[test]
    fn test_visit_pattern() {
        let config = default_config();
        let visitor = FormatVisitor::new(&config);

        // Wildcard
        let wildcard = visitor.visit_pattern(&Pattern::Wildcard);
        assert!(!wildcard.is_empty());

        // Simple binding
        let binding = Pattern::Binding {
            name: "x".to_string(),
            mutable: false,
        };
        let binding_doc = visitor.visit_pattern(&binding);
        assert!(!binding_doc.is_empty());

        // Mutable binding
        let mut_binding = Pattern::Binding {
            name: "y".to_string(),
            mutable: true,
        };
        let mut_doc = visitor.visit_pattern(&mut_binding);
        assert!(!mut_doc.is_empty());
    }

    #[test]
    fn test_visit_type() {
        let config = default_config();
        let visitor = FormatVisitor::new(&config);

        // Unit type
        let doc = visitor.visit_type(&TypeExpr::Unit);
        assert!(!doc.is_empty());

        // Reference type (mutable) - Sounio uses &!
        let ref_type = TypeExpr::Reference {
            mutable: true,
            inner: Box::new(TypeExpr::Named {
                path: Path::simple("i32"),
                args: vec![],
                unit: None,
            }),
        };
        let ref_doc = visitor.visit_type(&ref_type);
        assert!(!ref_doc.is_empty());
    }
}
