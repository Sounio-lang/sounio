//! Documentation extractor
//!
//! Extracts documentation from AST.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::ast::{
    self, Ast, EnumDef, FnDef, GenericParam, Generics, GlobalDef, ImplDef, Item, Param, StructDef,
    TraitDef, TraitFnDef, TraitItem, TypeAliasDef, TypeExpr, VariantData,
    Visibility as AstVisibility,
};

use super::model::*;
use super::parser::{self};

/// Extract documentation from a crate
pub struct DocExtractor {
    /// Crate name
    crate_name: String,

    /// Crate version
    version: String,

    /// Include private items
    document_private: bool,

    /// Include hidden items (#[doc(hidden)])
    document_hidden: bool,
}

impl DocExtractor {
    /// Create a new documentation extractor
    pub fn new(crate_name: &str, version: &str) -> Self {
        Self {
            crate_name: crate_name.to_string(),
            version: version.to_string(),
            document_private: false,
            document_hidden: false,
        }
    }

    /// Set whether to document private items
    pub fn document_private(mut self, yes: bool) -> Self {
        self.document_private = yes;
        self
    }

    /// Set whether to document hidden items
    pub fn document_hidden(mut self, yes: bool) -> Self {
        self.document_hidden = yes;
        self
    }

    /// Extract documentation from AST
    pub fn extract(&self, ast: &Ast) -> CrateDoc {
        let mut items = BTreeMap::new();
        let mut search_index = SearchIndex::new();

        // Extract root module
        let root_module = self.extract_module(ast, &self.crate_name, &mut items, &mut search_index);

        CrateDoc {
            name: self.crate_name.clone(),
            version: self.version.clone(),
            doc: None, // Would extract from //! comments at top of lib.sio
            root_module,
            items,
            search_index,
            source_files: Vec::new(),
        }
    }

    /// Extract documentation from AST with crate-level doc
    pub fn extract_with_crate_doc(&self, ast: &Ast, crate_doc: Option<String>) -> CrateDoc {
        let mut doc = self.extract(ast);
        doc.doc = crate_doc;
        doc
    }

    /// Extract a module
    fn extract_module(
        &self,
        ast: &Ast,
        path: &str,
        items: &mut BTreeMap<String, DocItem>,
        search_index: &mut SearchIndex,
    ) -> ModuleDoc {
        let name = path.rsplit("::").next().unwrap_or(path).to_string();

        let mut module = ModuleDoc::new(name.clone(), path.to_string());

        // Add module to search index
        search_index.add(SearchEntry {
            path: path.to_string(),
            name: name.clone(),
            kind: SearchKind::Module,
            desc: String::new(),
            parent: None,
        });

        for item in &ast.items {
            match item {
                Item::Function(f) => {
                    if self.should_document_visibility(&f.visibility) {
                        let func_doc = self.extract_function(f, path);

                        // Add to search index
                        search_index.add(SearchEntry {
                            path: func_doc.path.clone(),
                            name: func_doc.name.clone(),
                            kind: SearchKind::Function,
                            desc: self.get_summary(&func_doc.doc),
                            parent: Some(path.to_string()),
                        });

                        items.insert(func_doc.path.clone(), DocItem::Function(func_doc.clone()));
                        module.functions.push(func_doc);
                    }
                }

                Item::Struct(s) => {
                    if self.should_document_visibility(&s.visibility) {
                        let type_doc = self.extract_struct(s, path, search_index);
                        items.insert(type_doc.path.clone(), DocItem::Type(type_doc.clone()));
                        module.types.push(type_doc);
                    }
                }

                Item::Enum(e) => {
                    if self.should_document_visibility(&e.visibility) {
                        let type_doc = self.extract_enum(e, path, search_index);
                        items.insert(type_doc.path.clone(), DocItem::Type(type_doc.clone()));
                        module.types.push(type_doc);
                    }
                }

                Item::Trait(t) => {
                    if self.should_document_visibility(&t.visibility) {
                        let trait_doc = self.extract_trait(t, path, search_index);
                        items.insert(trait_doc.path.clone(), DocItem::Trait(trait_doc.clone()));
                        module.traits.push(trait_doc);
                    }
                }

                Item::TypeAlias(t) => {
                    if self.should_document_visibility(&t.visibility) {
                        let type_doc = self.extract_type_alias(t, path);

                        search_index.add(SearchEntry {
                            path: type_doc.path.clone(),
                            name: type_doc.name.clone(),
                            kind: SearchKind::TypeAlias,
                            desc: self.get_summary(&type_doc.doc),
                            parent: Some(path.to_string()),
                        });

                        items.insert(type_doc.path.clone(), DocItem::Type(type_doc.clone()));
                        module.types.push(type_doc);
                    }
                }

                Item::Global(g) => {
                    if g.is_const && self.should_document_visibility(&g.visibility) {
                        let const_doc = self.extract_constant(g, path);

                        search_index.add(SearchEntry {
                            path: const_doc.path.clone(),
                            name: const_doc.name.clone(),
                            kind: SearchKind::Constant,
                            desc: self.get_summary(&const_doc.doc),
                            parent: Some(path.to_string()),
                        });

                        items.insert(const_doc.path.clone(), DocItem::Constant(const_doc.clone()));
                        module.constants.push(const_doc);
                    }
                }

                Item::Impl(impl_def) => {
                    // Process impl blocks to add methods to types
                    self.process_impl(impl_def, path, items, search_index, &mut module);
                }

                Item::Import(import) => {
                    // Record re-exports
                    module.reexports.push(ReexportDoc {
                        name: import.path.name().unwrap_or("").to_string(),
                        original_path: import.path.to_string(),
                        doc: None,
                        visibility: Visibility::Public,
                    });
                }

                _ => {}
            }
        }

        module
    }

    /// Extract function documentation
    fn extract_function(&self, f: &FnDef, parent_path: &str) -> FunctionDoc {
        let path = format!("{}::{}", parent_path, f.name);
        let sig = self.render_function_signature(f);

        let mut func = FunctionDoc::new(f.name.clone(), path);
        func.visibility = self.convert_visibility(&f.visibility);
        func.signature = sig;
        func.is_unsafe = f.modifiers.is_unsafe;
        func.is_async = f.modifiers.is_async;
        func.is_kernel = f.modifiers.is_kernel;
        func.source = SourceLocation {
            file: PathBuf::new(),
            line: f.span.start as u32,
            column: 0,
        };

        // Extract type parameters
        func.type_params = self.extract_generics(&f.generics);

        // Extract parameters
        func.params = f.params.iter().map(|p| self.extract_param(p)).collect();

        // Extract return type
        if let Some(ref ret) = f.return_type {
            func.return_type = self.type_expr_to_info(ret);
        }

        // Extract effects
        func.effects = f.effects.iter().map(|e| e.name.to_string()).collect();

        // Extract where clauses
        func.where_clauses = f
            .where_clause
            .iter()
            .map(|w| WhereClause {
                ty: self.type_expr_to_string(&w.ty),
                bounds: w.bounds.iter().map(|b| b.to_string()).collect(),
            })
            .collect();

        func
    }

    /// Render function signature as string
    fn render_function_signature(&self, f: &FnDef) -> String {
        let mut sig = String::new();

        // Visibility
        match f.visibility {
            AstVisibility::Public => sig.push_str("pub "),
            AstVisibility::Private => {}
        }

        // Modifiers
        if f.modifiers.is_unsafe {
            sig.push_str("unsafe ");
        }
        if f.modifiers.is_async {
            sig.push_str("async ");
        }
        if f.modifiers.is_kernel {
            sig.push_str("kernel ");
        }

        sig.push_str("fn ");
        sig.push_str(&f.name);

        // Type parameters
        if !f.generics.params.is_empty() {
            sig.push('<');
            for (i, param) in f.generics.params.iter().enumerate() {
                if i > 0 {
                    sig.push_str(", ");
                }
                match param {
                    GenericParam::Type { name, bounds, .. } => {
                        sig.push_str(name);
                        if !bounds.is_empty() {
                            sig.push_str(": ");
                            sig.push_str(
                                &bounds
                                    .iter()
                                    .map(|b| b.to_string())
                                    .collect::<Vec<_>>()
                                    .join(" + "),
                            );
                        }
                    }
                    GenericParam::Const { name, ty } => {
                        sig.push_str("const ");
                        sig.push_str(name);
                        sig.push_str(": ");
                        sig.push_str(&self.type_expr_to_string(ty));
                    }
                }
            }
            sig.push('>');
        }

        // Parameters
        sig.push('(');
        for (i, p) in f.params.iter().enumerate() {
            if i > 0 {
                sig.push_str(", ");
            }
            if p.is_mut {
                sig.push_str("mut ");
            }
            sig.push_str(&self.pattern_to_string(&p.pattern));
            sig.push_str(": ");
            sig.push_str(&self.type_expr_to_string(&p.ty));
        }
        sig.push(')');

        // Return type
        if let Some(ref ret) = f.return_type {
            sig.push_str(" -> ");
            sig.push_str(&self.type_expr_to_string(ret));
        }

        // Effects
        if !f.effects.is_empty() {
            sig.push_str(" with ");
            sig.push_str(
                &f.effects
                    .iter()
                    .map(|e| e.name.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }

        sig
    }

    /// Extract struct documentation
    fn extract_struct(
        &self,
        s: &StructDef,
        parent_path: &str,
        search_index: &mut SearchIndex,
    ) -> TypeDoc {
        let path = format!("{}::{}", parent_path, s.name);

        let mut type_doc = TypeDoc::new(s.name.clone(), path.clone(), TypeKind::Struct);
        type_doc.visibility = self.convert_visibility(&s.visibility);
        type_doc.type_params = self.extract_generics(&s.generics);
        type_doc.modifiers = TypeModifiers {
            linear: s.modifiers.linear,
            affine: s.modifiers.affine,
        };
        type_doc.source = SourceLocation {
            file: PathBuf::new(),
            line: s.span.start as u32,
            column: 0,
        };

        // Extract fields
        type_doc.fields = s
            .fields
            .iter()
            .map(|f| {
                let field_doc = FieldDoc {
                    name: f.name.clone(),
                    ty: self.type_expr_to_info(&f.ty),
                    doc: None,
                    visibility: self.convert_visibility(&f.visibility),
                };

                // Index field
                search_index.add(SearchEntry {
                    path: format!("{}::{}", path, f.name),
                    name: f.name.clone(),
                    kind: SearchKind::Field,
                    desc: String::new(),
                    parent: Some(path.clone()),
                });

                field_doc
            })
            .collect();

        // Index struct
        search_index.add(SearchEntry {
            path: path.clone(),
            name: s.name.clone(),
            kind: SearchKind::Struct,
            desc: self.get_summary(&type_doc.doc),
            parent: Some(parent_path.to_string()),
        });

        type_doc
    }

    /// Extract enum documentation
    fn extract_enum(
        &self,
        e: &EnumDef,
        parent_path: &str,
        search_index: &mut SearchIndex,
    ) -> TypeDoc {
        let path = format!("{}::{}", parent_path, e.name);

        let mut type_doc = TypeDoc::new(e.name.clone(), path.clone(), TypeKind::Enum);
        type_doc.visibility = self.convert_visibility(&e.visibility);
        type_doc.type_params = self.extract_generics(&e.generics);
        type_doc.modifiers = TypeModifiers {
            linear: e.modifiers.linear,
            affine: e.modifiers.affine,
        };
        type_doc.source = SourceLocation {
            file: PathBuf::new(),
            line: e.span.start as u32,
            column: 0,
        };

        // Extract variants
        type_doc.variants = e
            .variants
            .iter()
            .map(|v| {
                let variant_doc = VariantDoc {
                    name: v.name.clone(),
                    doc: None,
                    fields: match &v.data {
                        VariantData::Unit => Vec::new(),
                        VariantData::Tuple(types) => types
                            .iter()
                            .enumerate()
                            .map(|(i, ty)| FieldDoc {
                                name: i.to_string(),
                                ty: self.type_expr_to_info(ty),
                                doc: None,
                                visibility: Visibility::Public,
                            })
                            .collect(),
                        VariantData::Struct(fields) => fields
                            .iter()
                            .map(|f| FieldDoc {
                                name: f.name.clone(),
                                ty: self.type_expr_to_info(&f.ty),
                                doc: None,
                                visibility: self.convert_visibility(&f.visibility),
                            })
                            .collect(),
                    },
                    discriminant: None,
                };

                // Index variant
                search_index.add(SearchEntry {
                    path: format!("{}::{}", path, v.name),
                    name: v.name.clone(),
                    kind: SearchKind::Variant,
                    desc: String::new(),
                    parent: Some(path.clone()),
                });

                variant_doc
            })
            .collect();

        // Index enum
        search_index.add(SearchEntry {
            path: path.clone(),
            name: e.name.clone(),
            kind: SearchKind::Enum,
            desc: self.get_summary(&type_doc.doc),
            parent: Some(parent_path.to_string()),
        });

        type_doc
    }

    /// Extract trait documentation
    fn extract_trait(
        &self,
        t: &TraitDef,
        parent_path: &str,
        search_index: &mut SearchIndex,
    ) -> TraitDoc {
        let path = format!("{}::{}", parent_path, t.name);

        let mut trait_doc = TraitDoc::new(t.name.clone(), path.clone());
        trait_doc.visibility = self.convert_visibility(&t.visibility);
        trait_doc.type_params = self.extract_generics(&t.generics);
        trait_doc.super_traits = t.supertraits.iter().map(|s| s.to_string()).collect();
        trait_doc.source = SourceLocation {
            file: PathBuf::new(),
            line: t.span.start as u32,
            column: 0,
        };

        // Extract trait items
        for item in &t.items {
            match item {
                TraitItem::Fn(f) => {
                    let method_doc = self.extract_trait_fn(f, &path);

                    // Index method
                    search_index.add(SearchEntry {
                        path: method_doc.path.clone(),
                        name: method_doc.name.clone(),
                        kind: SearchKind::Method,
                        desc: self.get_summary(&method_doc.doc),
                        parent: Some(path.clone()),
                    });

                    if f.default_body.is_some() {
                        trait_doc.provided_methods.push(method_doc);
                    } else {
                        trait_doc.required_methods.push(method_doc);
                    }
                }
                TraitItem::Type(t) => {
                    trait_doc.assoc_types.push(AssocTypeDoc {
                        name: t.name.clone(),
                        doc: None,
                        bounds: t.bounds.iter().map(|b| b.to_string()).collect(),
                        default: t.default.as_ref().map(|d| self.type_expr_to_info(d)),
                    });
                }
            }
        }

        // Index trait
        search_index.add(SearchEntry {
            path: path.clone(),
            name: t.name.clone(),
            kind: SearchKind::Trait,
            desc: self.get_summary(&trait_doc.doc),
            parent: Some(parent_path.to_string()),
        });

        trait_doc
    }

    /// Extract trait function documentation
    fn extract_trait_fn(&self, f: &TraitFnDef, parent_path: &str) -> FunctionDoc {
        let path = format!("{}::{}", parent_path, f.name);

        let mut func = FunctionDoc::new(f.name.clone(), path);
        func.visibility = Visibility::Public;
        func.type_params = self.extract_generics(&f.generics);
        func.params = f.params.iter().map(|p| self.extract_param(p)).collect();

        if let Some(ref ret) = f.return_type {
            func.return_type = self.type_expr_to_info(ret);
        }

        func.effects = f.effects.iter().map(|e| e.name.to_string()).collect();

        // Build signature
        func.signature = self.render_trait_fn_signature(f);

        func
    }

    /// Render trait function signature
    fn render_trait_fn_signature(&self, f: &TraitFnDef) -> String {
        let mut sig = String::from("fn ");
        sig.push_str(&f.name);

        if !f.generics.params.is_empty() {
            sig.push('<');
            for (i, param) in f.generics.params.iter().enumerate() {
                if i > 0 {
                    sig.push_str(", ");
                }
                if let GenericParam::Type { name, .. } = param {
                    sig.push_str(name);
                }
            }
            sig.push('>');
        }

        sig.push('(');
        for (i, p) in f.params.iter().enumerate() {
            if i > 0 {
                sig.push_str(", ");
            }
            sig.push_str(&self.pattern_to_string(&p.pattern));
            sig.push_str(": ");
            sig.push_str(&self.type_expr_to_string(&p.ty));
        }
        sig.push(')');

        if let Some(ref ret) = f.return_type {
            sig.push_str(" -> ");
            sig.push_str(&self.type_expr_to_string(ret));
        }

        if !f.effects.is_empty() {
            sig.push_str(" with ");
            sig.push_str(
                &f.effects
                    .iter()
                    .map(|e| e.name.to_string())
                    .collect::<Vec<_>>()
                    .join(", "),
            );
        }

        sig
    }

    /// Extract type alias documentation
    fn extract_type_alias(&self, t: &TypeAliasDef, parent_path: &str) -> TypeDoc {
        let path = format!("{}::{}", parent_path, t.name);

        let mut type_doc = TypeDoc::new(t.name.clone(), path, TypeKind::TypeAlias);
        type_doc.visibility = self.convert_visibility(&t.visibility);
        type_doc.type_params = self.extract_generics(&t.generics);
        type_doc.source = SourceLocation {
            file: PathBuf::new(),
            line: t.span.start as u32,
            column: 0,
        };

        type_doc
    }

    /// Extract constant documentation
    fn extract_constant(&self, g: &GlobalDef, parent_path: &str) -> ConstantDoc {
        let name = self.pattern_to_string(&g.pattern);
        let path = format!("{}::{}", parent_path, name);

        ConstantDoc {
            name,
            path,
            doc: None,
            visibility: self.convert_visibility(&g.visibility),
            ty: g
                .ty
                .as_ref()
                .map(|t| self.type_expr_to_info(t))
                .unwrap_or(TypeInfo::unit()),
            value: Some(self.expr_to_string(&g.value)),
            source: SourceLocation {
                file: PathBuf::new(),
                line: g.span.start as u32,
                column: 0,
            },
        }
    }

    /// Process impl block
    fn process_impl(
        &self,
        impl_def: &ImplDef,
        parent_path: &str,
        items: &mut BTreeMap<String, DocItem>,
        search_index: &mut SearchIndex,
        module: &mut ModuleDoc,
    ) {
        let target_type = self.type_expr_to_string(&impl_def.target_type);
        let type_path = format!("{}::{}", parent_path, target_type);

        // Find the type in module.types and add methods to it
        for type_doc in &mut module.types {
            if type_doc.path == type_path || type_doc.name == target_type {
                // Check if this is a trait impl
                if let Some(ref trait_ref) = impl_def.trait_ref {
                    let trait_impl = TraitImplDoc {
                        trait_path: trait_ref.to_string(),
                        type_params: self.extract_generics(&impl_def.generics),
                        where_clauses: Vec::new(),
                        is_blanket: false,
                        methods: Vec::new(),
                        assoc_types: Vec::new(),
                    };
                    type_doc.trait_impls.push(trait_impl);
                }

                // Add methods
                for impl_item in &impl_def.items {
                    if let ast::ImplItem::Fn(f) = impl_item
                        && self.should_document_visibility(&f.visibility)
                    {
                        let method_doc = self.extract_function(f, &type_path);

                        search_index.add(SearchEntry {
                            path: method_doc.path.clone(),
                            name: method_doc.name.clone(),
                            kind: SearchKind::Method,
                            desc: self.get_summary(&method_doc.doc),
                            parent: Some(type_path.clone()),
                        });

                        type_doc.methods.push(method_doc);
                    }
                }

                break;
            }
        }
    }

    /// Extract generics
    fn extract_generics(&self, generics: &Generics) -> Vec<TypeParamInfo> {
        generics
            .params
            .iter()
            .filter_map(|p| {
                match p {
                    GenericParam::Type {
                        name,
                        bounds,
                        default,
                    } => Some(TypeParamInfo {
                        name: name.clone(),
                        bounds: bounds.iter().map(|b| b.to_string()).collect(),
                        default: default.as_ref().map(|d| self.type_expr_to_info(d)),
                    }),
                    GenericParam::Const { .. } => None, // Handle const generics separately if needed
                }
            })
            .collect()
    }

    /// Extract parameter info
    fn extract_param(&self, p: &Param) -> ParamInfo {
        let name = self.pattern_to_string(&p.pattern);
        let is_self = name == "self";

        ParamInfo {
            name,
            ty: self.type_expr_to_info(&p.ty),
            is_self,
            self_kind: if is_self {
                Some(self.infer_self_kind(&p.ty))
            } else {
                None
            },
            is_mut: p.is_mut,
        }
    }

    /// Infer self kind from type
    fn infer_self_kind(&self, ty: &TypeExpr) -> SelfKind {
        match ty {
            TypeExpr::Reference { mutable: false, .. } => SelfKind::Ref,
            TypeExpr::Reference { mutable: true, .. } => SelfKind::RefMut,
            _ => SelfKind::Value,
        }
    }

    /// Convert AST visibility to doc visibility
    fn convert_visibility(&self, vis: &AstVisibility) -> Visibility {
        match vis {
            AstVisibility::Public => Visibility::Public,
            AstVisibility::Private => Visibility::Private,
        }
    }

    /// Check if an item should be documented based on visibility
    fn should_document_visibility(&self, vis: &AstVisibility) -> bool {
        match vis {
            AstVisibility::Public => true,
            AstVisibility::Private => self.document_private,
        }
    }

    /// Get summary from documentation
    fn get_summary(&self, doc: &Option<String>) -> String {
        doc.as_ref()
            .and_then(|d| parser::parse_sections(d).summary)
            .unwrap_or_default()
    }

    /// Convert type expression to TypeInfo
    fn type_expr_to_info(&self, ty: &TypeExpr) -> TypeInfo {
        TypeInfo {
            display: self.type_expr_to_string(ty),
            links: Vec::new(), // Would resolve links to other types
        }
    }

    /// Convert type expression to string
    fn type_expr_to_string(&self, ty: &TypeExpr) -> String {
        match ty {
            TypeExpr::Unit => "unit".to_string(),
            TypeExpr::SelfType => "Self".to_string(),
            TypeExpr::Named { path, args, unit } => {
                let mut s = path.to_string();
                if !args.is_empty() {
                    s.push('<');
                    s.push_str(
                        &args
                            .iter()
                            .map(|a| self.type_expr_to_string(a))
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                    s.push('>');
                }
                if let Some(u) = unit {
                    s.push('<');
                    s.push_str(u);
                    s.push('>');
                }
                s
            }
            TypeExpr::Reference { mutable, inner } => {
                if *mutable {
                    format!("&!{}", self.type_expr_to_string(inner))
                } else {
                    format!("&{}", self.type_expr_to_string(inner))
                }
            }
            TypeExpr::RawPointer { mutable, inner } => {
                if *mutable {
                    format!("*mut {}", self.type_expr_to_string(inner))
                } else {
                    format!("*const {}", self.type_expr_to_string(inner))
                }
            }
            TypeExpr::Array { element, size } => {
                if let Some(size) = size {
                    format!(
                        "[{}; {}]",
                        self.type_expr_to_string(element),
                        self.expr_to_string(size)
                    )
                } else {
                    format!("[{}]", self.type_expr_to_string(element))
                }
            }
            TypeExpr::Tuple(types) => {
                format!(
                    "({})",
                    types
                        .iter()
                        .map(|t| self.type_expr_to_string(t))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            TypeExpr::Function {
                params,
                return_type,
                effects,
            } => {
                let mut s = format!(
                    "fn({})",
                    params
                        .iter()
                        .map(|p| self.type_expr_to_string(p))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                s.push_str(" -> ");
                s.push_str(&self.type_expr_to_string(return_type));
                if !effects.is_empty() {
                    s.push_str(" with ");
                    s.push_str(
                        &effects
                            .iter()
                            .map(|e| e.name.to_string())
                            .collect::<Vec<_>>()
                            .join(", "),
                    );
                }
                s
            }
            TypeExpr::Infer => "_".to_string(),

            // Epistemic types
            TypeExpr::Knowledge {
                value_type,
                epsilon,
                validity,
                provenance,
            } => {
                let mut s = format!("Knowledge[{}", self.type_expr_to_string(value_type));
                // Epsilon, validity, provenance would be added here if needed
                s.push(']');
                s
            }
            TypeExpr::Quantity { numeric_type, unit } => {
                let unit_str = unit
                    .base_units
                    .iter()
                    .map(|(name, exp)| {
                        if *exp == 1 {
                            name.clone()
                        } else {
                            format!("{}^{}", name, exp)
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("*");
                format!(
                    "Quantity[{}, {}]",
                    self.type_expr_to_string(numeric_type),
                    unit_str
                )
            }
            TypeExpr::Tensor {
                element_type,
                shape,
            } => {
                let shape_str = shape
                    .iter()
                    .map(|d| match d {
                        ast::TensorDim::Named(n) => n.clone(),
                        ast::TensorDim::Fixed(s) => s.to_string(),
                        ast::TensorDim::Dynamic => "_".to_string(),
                        ast::TensorDim::Expr(_) => "<expr>".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "Tensor[{}, ({})]",
                    self.type_expr_to_string(element_type),
                    shape_str
                )
            }
            TypeExpr::Ontology { ontology, term } => {
                if let Some(t) = term {
                    format!("OntologyTerm[{}:{}]", ontology, t)
                } else {
                    format!("OntologyTerm[{}]", ontology)
                }
            }
            TypeExpr::Linear { inner, linearity } => {
                let kind = match linearity {
                    ast::LinearityKind::Linear => "linear",
                    ast::LinearityKind::Affine => "affine",
                    ast::LinearityKind::Relevant => "relevant",
                    ast::LinearityKind::Unrestricted => "",
                };
                if kind.is_empty() {
                    self.type_expr_to_string(inner)
                } else {
                    format!("{} @ {}", self.type_expr_to_string(inner), kind)
                }
            }
            TypeExpr::Effected { inner, effects } => {
                let effects_str = effects.effects.join(", ");
                format!("{} ! {{{}}}", self.type_expr_to_string(inner), effects_str)
            }
            TypeExpr::Tile {
                element_type,
                tile_m,
                tile_n,
                layout,
            } => {
                let mut s = format!(
                    "tile<{}, {}, {}>",
                    self.type_expr_to_string(element_type),
                    tile_m,
                    tile_n
                );
                if let Some(l) = layout {
                    s = format!(
                        "tile<{}, {}, {}, \"{}\">",
                        self.type_expr_to_string(element_type),
                        tile_m,
                        tile_n,
                        l
                    );
                }
                s
            }
            TypeExpr::Refinement {
                var,
                base_type,
                predicate,
            } => {
                format!(
                    "{{ {}: {} | {} }}",
                    var,
                    self.type_expr_to_string(base_type),
                    self.expr_to_string(predicate)
                )
            }
        }
    }

    /// Convert pattern to string
    fn pattern_to_string(&self, pattern: &ast::Pattern) -> String {
        match pattern {
            ast::Pattern::Wildcard => "_".to_string(),
            ast::Pattern::Binding { name, mutable } => {
                if *mutable {
                    format!("mut {}", name)
                } else {
                    name.clone()
                }
            }
            ast::Pattern::Literal(lit) => self.literal_to_string(lit),
            ast::Pattern::Tuple(patterns) => {
                format!(
                    "({})",
                    patterns
                        .iter()
                        .map(|p| self.pattern_to_string(p))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ast::Pattern::Struct { path, fields } => {
                format!(
                    "{} {{ {} }}",
                    path,
                    fields
                        .iter()
                        .map(|(n, p)| format!("{}: {}", n, self.pattern_to_string(p)))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            ast::Pattern::Enum { path, patterns } => {
                if let Some(patterns) = patterns {
                    format!(
                        "{}({})",
                        path,
                        patterns
                            .iter()
                            .map(|p| self.pattern_to_string(p))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    path.to_string()
                }
            }
            ast::Pattern::Or(patterns) => patterns
                .iter()
                .map(|p| self.pattern_to_string(p))
                .collect::<Vec<_>>()
                .join(" | "),
        }
    }

    /// Convert literal to string
    fn literal_to_string(&self, lit: &ast::Literal) -> String {
        match lit {
            ast::Literal::Unit => "()".to_string(),
            ast::Literal::Bool(b) => b.to_string(),
            ast::Literal::Int(i) => i.to_string(),
            ast::Literal::Float(f) => f.to_string(),
            ast::Literal::Char(c) => format!("'{}'", c),
            ast::Literal::String(s) => format!("\"{}\"", s),
            ast::Literal::IntUnit(i, u) => format!("{}_{}", i, u),
            ast::Literal::FloatUnit(f, u) => format!("{}_{}", f, u),
        }
    }

    /// Convert expression to string (simplified)
    fn expr_to_string(&self, expr: &ast::Expr) -> String {
        match expr {
            ast::Expr::Literal { value, .. } => self.literal_to_string(value),
            ast::Expr::Path { path, .. } => path.to_string(),
            ast::Expr::Binary {
                op, left, right, ..
            } => {
                format!(
                    "{} {} {}",
                    self.expr_to_string(left),
                    self.binary_op_to_string(op),
                    self.expr_to_string(right)
                )
            }
            ast::Expr::Unary { op, expr, .. } => {
                format!(
                    "{}{}",
                    self.unary_op_to_string(op),
                    self.expr_to_string(expr)
                )
            }
            ast::Expr::Call { callee, args, .. } => {
                format!(
                    "{}({})",
                    self.expr_to_string(callee),
                    args.iter()
                        .map(|a| self.expr_to_string(a))
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            _ => "...".to_string(),
        }
    }

    /// Convert binary operator to string
    fn binary_op_to_string(&self, op: &ast::BinaryOp) -> &'static str {
        match op {
            ast::BinaryOp::Add => "+",
            ast::BinaryOp::Sub => "-",
            ast::BinaryOp::Mul => "*",
            ast::BinaryOp::Div => "/",
            ast::BinaryOp::Rem => "%",
            ast::BinaryOp::Eq => "==",
            ast::BinaryOp::Ne => "!=",
            ast::BinaryOp::Lt => "<",
            ast::BinaryOp::Le => "<=",
            ast::BinaryOp::Gt => ">",
            ast::BinaryOp::Ge => ">=",
            ast::BinaryOp::And => "&&",
            ast::BinaryOp::Or => "||",
            ast::BinaryOp::BitAnd => "&",
            ast::BinaryOp::BitOr => "|",
            ast::BinaryOp::BitXor => "^",
            ast::BinaryOp::Shl => "<<",
            ast::BinaryOp::Shr => ">>",
            ast::BinaryOp::PlusMinus => "±",
            ast::BinaryOp::Concat => "++",
        }
    }

    /// Convert unary operator to string
    fn unary_op_to_string(&self, op: &ast::UnaryOp) -> &'static str {
        match op {
            ast::UnaryOp::Neg => "-",
            ast::UnaryOp::Not => "!",
            ast::UnaryOp::Ref => "&",
            ast::UnaryOp::RefMut => "&!",
            ast::UnaryOp::Deref => "*",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extractor_new() {
        let extractor = DocExtractor::new("test_crate", "0.1.0");
        assert_eq!(extractor.crate_name, "test_crate");
        assert_eq!(extractor.version, "0.1.0");
        assert!(!extractor.document_private);
    }

    #[test]
    fn test_extractor_document_private() {
        let extractor = DocExtractor::new("test", "0.1.0").document_private(true);
        assert!(extractor.document_private);
    }
}
