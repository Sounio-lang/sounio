//! Name resolution pass

// Miette's derive macros generate code that triggers unused_assignments warnings
#![allow(unused_assignments)]

use super::module_tree::{
    ImportEntry, ItemKind, ModuleId as TreeModuleId, ModuleItem, ModuleTree,
};
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

    #[error("Unresolved import: {path}")]
    #[diagnostic(help("make sure the module exists and is accessible"))]
    UnresolvedImport {
        path: String,
        reason: String,
        #[label("{reason}")]
        span: SourceSpan,
    },

    #[error("Cannot access private item `{name}` in module `{module_path}`")]
    #[diagnostic(help("consider making the item public with `pub`"))]
    PrivateItem {
        name: String,
        module_path: String,
        #[label("this item is private")]
        span: SourceSpan,
    },
}

/// Resolved AST (AST + symbol table + module tree)
#[derive(Debug)]
pub struct ResolvedAst {
    pub ast: Ast,
    pub symbols: SymbolTable,
    /// Module tree with visibility-aware item lookup
    pub module_tree: ModuleTree,
}

/// Resolve names in an AST
pub fn resolve(ast: Ast) -> Result<ResolvedAst> {
    let resolver = Resolver::new();
    resolver.resolve(ast)
}

/// Pending unresolved import for later verification
#[derive(Debug, Clone)]
struct PendingImport {
    path: Vec<String>,
    reason: String,
    span: Span,
}

/// Name resolver
pub struct Resolver {
    symbols: SymbolTable,
    errors: Vec<ResolveError>,
    /// Module tree being built
    module_tree: ModuleTree,
    /// Current module path for tree building
    current_tree_module: TreeModuleId,
    /// Mapping from NodeId to DefId for items resolved through ModuleTree imports
    /// This is populated after resolve_imports() and used during second pass
    import_node_to_def: std::collections::HashMap<NodeId, DefId>,
    /// Imports that failed to resolve initially - verified after all modules loaded
    unresolved_imports: Vec<PendingImport>,
}

impl Resolver {
    pub fn new() -> Self {
        let mut resolver = Self {
            symbols: SymbolTable::new(),
            errors: Vec::new(),
            module_tree: ModuleTree::new(),
            current_tree_module: TreeModuleId::root(),
            import_node_to_def: std::collections::HashMap::new(),
            unresolved_imports: Vec::new(),
        };
        // Register compiler intrinsics/builtins
        resolver.register_builtins();
        resolver
    }

    /// Register compiler builtin functions
    fn register_builtins(&mut self) {
        let builtins = [
            // Slice construction intrinsics
            "__builtin_slice_from_raw_parts",
            "__builtin_slice_from_raw_parts_mut",
            // I/O builtins
            "print", "println", "dbg", "panic", "assert", "assert_eq",
            "read_line", "format", "to_string",
            // Type introspection
            "len", "type_of", "parse_int", "parse_float",
            // Math builtins
            "sqrt", "abs", "sin", "cos", "tan", "exp", "log", "pow",
            "floor", "ceil", "round", "min", "max",
            // Linear algebra
            "vec2", "vec3", "vec4", "mat2", "mat3", "mat4", "quat",
            "dot", "cross", "normalize", "length", "length_squared",
            "quat_mul", "quat_conj", "quat_inv", "quat_normalize", "quat_identity",
            "mat_mul", "transpose", "inverse", "determinant",
            "lerp", "slerp",
            "quat_to_euler", "euler_to_quat", "quat_to_mat3", "quat_to_mat4", "mat3_to_quat",
            "hamilton_product", "quat_rotate_vec", "quat_score", "quat_embed_init",
            "quat_normalize_embed", "quat_inner_product",
            // Dual number operations (forward-mode autodiff)
            "dual", "dual_value", "dual_deriv",
            "dual_add", "dual_sub", "dual_mul", "dual_div",
            "dual_sin", "dual_cos", "dual_exp", "dual_log", "dual_sqrt", "dual_pow",
            "dual_tan", "dual_atan", "dual_abs",
            "dual_asin", "dual_acos", "dual_sinh", "dual_cosh", "dual_tanh",
            "dual_asinh", "dual_acosh", "dual_atanh",
            "dual_log2", "dual_log10", "dual_atan2",
            // Autodiff higher-order functions
            "grad", "jacobian", "hessian",
            // Array operations
            "array_len", "array_ptr",
            // Pointer operations
            "ptr_load_f64", "ptr_store_f64",
            // FFI / Raw pointer operations
            "null_ptr", "null_mut", "is_null", "ptr_eq", "ptr_addr",
            "ptr_from_addr", "ptr_from_addr_mut",
            "ptr_offset", "ptr_add", "ptr_sub", "ptr_diff",
            "as_const", "as_mut", "size_of", "align_of",
            // Option/Result constructors
            "Some", "None", "Ok", "Err",
        ];

        for name in builtins {
            let def_id = self.symbols.fresh_def_id();
            let _ = self.symbols.define(name.to_string(), def_id);
            self.symbols.insert(Symbol {
                def_id,
                name: name.to_string(),
                kind: DefKind::Function,
                node_id: NodeId(0),
                span: Span::default(),
                parent: None,
            });
        }
    }

    /// Build mappings from resolved imports to DefIds
    ///
    /// After ModuleTree.resolve_imports() completes, this method:
    /// 1. Walks all modules in the tree
    /// 2. For each resolved import, finds or creates the DefId for the imported symbol
    /// 3. Registers the import in the symbol table scope
    fn build_import_mappings(&mut self) {
        // Collect all module IDs first to avoid borrow issues
        let module_ids: Vec<TreeModuleId> = self.module_tree.all_module_ids().cloned().collect();

        for module_id in module_ids {
            // Get visible items for this module (includes resolved imports)
            let visible = self.module_tree.visible_items_in(&module_id);

            for (name, node_id) in visible {
                // Skip error sentinels
                if node_id.0 == u32::MAX {
                    continue;
                }

                // Check if we already have a DefId for this NodeId
                if let Some(def_id) = self.symbols.def_for_node(node_id) {
                    // Map the import's resolved NodeId to the existing DefId
                    self.import_node_to_def.insert(node_id, def_id);
                }
            }
        }
    }

    /// Look up a name in the current module, checking imports from ModuleTree
    ///
    /// Returns the DefId if the name is:
    /// 1. Defined in the current lexical scope
    /// 2. Imported into the current module via use statement
    fn lookup_with_imports(&self, name: &str) -> Option<DefId> {
        // First check lexical scopes (locals, function params, etc.)
        if let Some(def_id) = self.symbols.lookup(name) {
            return Some(def_id);
        }

        // Then check imports in the current module tree module
        if let Some(module) = self.module_tree.get(&self.current_tree_module) {
            // Check resolved imports
            for import in &module.imports {
                if import.local_name == name {
                    if let Some(node_id) = import.resolved {
                        // Skip error sentinels
                        if node_id.0 != u32::MAX {
                            // Look up the DefId for this NodeId
                            if let Some(def_id) = self.symbols.def_for_node(node_id) {
                                return Some(def_id);
                            }
                            // Also check our cached mapping
                            if let Some(def_id) = self.import_node_to_def.get(&node_id) {
                                return Some(*def_id);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Look up a type name in the current module, checking imports from ModuleTree
    fn lookup_type_with_imports(&self, name: &str) -> Option<DefId> {
        // First check lexical scopes
        if let Some(def_id) = self.symbols.lookup_type(name) {
            return Some(def_id);
        }

        // Then check imports in the current module tree module
        if let Some(module) = self.module_tree.get(&self.current_tree_module) {
            for import in &module.imports {
                if import.local_name == name {
                    if let Some(node_id) = import.resolved {
                        if node_id.0 != u32::MAX {
                            if let Some(def_id) = self.symbols.def_for_node(node_id) {
                                return Some(def_id);
                            }
                            if let Some(def_id) = self.import_node_to_def.get(&node_id) {
                                return Some(*def_id);
                            }
                        }
                    }
                }
            }
        }

        None
    }

    /// Resolve a multi-segment path (e.g., foo::Bar or std::collections::HashMap)
    ///
    /// Returns the DefId if the path resolves to a valid item.
    /// Records a PrivateItem error if the item exists but is not visible.
    fn resolve_path_segments(&mut self, segments: &[String]) -> Option<DefId> {
        if segments.is_empty() {
            return None;
        }

        if segments.len() == 1 {
            // Single segment - use normal lookup
            return self.lookup_with_imports(&segments[0]);
        }

        // Multi-segment path: could be a module path or type::variant
        // First segment might be a module name, imported module, or an imported type
        let first = &segments[0];

        // First, check if this is a Type::Variant pattern (e.g., Color::Red)
        // by checking if first segment resolves to a type (enum)
        if segments.len() == 2 {
            if let Some(type_def_id) = self.lookup_type_with_imports(&segments[0]) {
                // First segment is a type - check if it's an enum with this variant
                if let Some(symbol) = self.symbols.get(type_def_id) {
                    if matches!(symbol.kind, DefKind::Enum { .. }) {
                        // Look up the variant in the enum
                        let variant_name = &segments[1];
                        // Try to find a symbol for this variant under the enum
                        if let Some(variant_id) = self.symbols.lookup_variant(type_def_id, variant_name) {
                            return Some(variant_id);
                        }
                    }
                }
            }
        }

        // Try to find the starting module
        let start_module = if let Some(module) = self.module_tree.get(&self.current_tree_module) {
            // Check if first segment is a child module
            if let Some(child_id) = module.children.get(first) {
                Some(child_id.clone())
            }
            // Check if first segment is an imported module
            else if let Some(import) = module.imports.iter().find(|i| i.local_name == *first) {
                // The import might point to a module
                Some(TreeModuleId(import.source_path.clone()))
            } else {
                // Try from root
                let root_child = TreeModuleId::root().join(first);
                if self.module_tree.contains(&root_child) {
                    Some(root_child)
                } else {
                    None
                }
            }
        } else {
            None
        };

        let start_module = start_module?;

        // Walk the remaining path
        let mut current_module = start_module;
        for (i, segment) in segments[1..].iter().enumerate() {
            let is_last = i == segments.len() - 2;

            if is_last {
                // Last segment is the item name
                let module = self.module_tree.get(&current_module)?;
                let module_path = current_module.to_string();
                let from_module = self.current_tree_module.clone();

                // Check items with visibility enforcement
                if let Some(item) = module.items.iter().find(|item| item.name == *segment) {
                    // Check visibility - private items not accessible from outside
                    if !module.is_visible(item, &from_module) {
                        self.errors.push(ResolveError::PrivateItem {
                            name: segment.clone(),
                            module_path,
                            span: SourceSpan::from(0..1),
                        });
                        return None;
                    }
                    return self.symbols.def_for_node(item.node_id);
                }

                // Check resolved imports (for re-exports) - re-exports are public by definition
                for import in &module.imports {
                    if import.local_name == *segment {
                        if let Some(node_id) = import.resolved {
                            if node_id.0 != u32::MAX {
                                if let Some(def_id) = self.symbols.def_for_node(node_id) {
                                    return Some(def_id);
                                }
                            }
                        }
                    }
                }

                return None;
            } else {
                // Intermediate segment is a module name
                let module = self.module_tree.get(&current_module)?;
                current_module = module.children.get(segment)?.clone();
            }
        }

        None
    }

    /// Resolve a multi-segment type path
    fn resolve_type_path_segments(&mut self, segments: &[String]) -> Option<DefId> {
        // For types, we use the same path resolution logic
        self.resolve_path_segments(segments)
    }

    /// Verify unresolved imports after all modules have been loaded
    ///
    /// Some imports fail initially because they reference modules that haven't
    /// been loaded yet. This method re-checks those imports and reports errors
    /// for any that still cannot be resolved.
    fn verify_unresolved_imports(&mut self) {
        // Take ownership of unresolved imports to avoid borrow issues
        let pending = std::mem::take(&mut self.unresolved_imports);

        for import in pending {
            // Try to resolve again - module may have been loaded
            let result = self.symbols.process_import(
                &import.path,
                None, // Already processed items
                false,
            );

            if result.is_err() {
                // Still unresolved - check if module exists in module_tree
                let module_id = TreeModuleId(import.path.clone());

                // Skip stdlib imports as they may be external
                if import.path.first().map(|s| s.as_str()) == Some("std") {
                    continue;
                }

                // Only report error if module doesn't exist at all
                if !self.module_tree.contains(&module_id) {
                    self.errors.push(ResolveError::UnresolvedImport {
                        path: import.path.join("::"),
                        reason: import.reason,
                        span: self.span_to_source(import.span),
                    });
                }
            }
        }
    }

    /// Resolve all names in the AST
    pub fn resolve(mut self, ast: Ast) -> Result<ResolvedAst> {
        // First pass: collect all top-level definitions and build module tree
        for item in &ast.items {
            self.collect_item(item);
        }

        // Resolve imports in the module tree (multi-pass algorithm)
        // This handles forward references, transitive imports, and detects cycles
        if let Err(import_errors) = self.module_tree.resolve_imports() {
            // Convert import errors to resolution errors for reporting
            // We don't fail here because some imports may be to external modules
            // that will be loaded later. Instead, we log warnings.
            for err in import_errors {
                // Only report module-not-found as errors for now
                // Unresolved imports to existing modules might be typos
                match &err {
                    super::module_tree::ImportError::NotVisible { name, path } => {
                        // This is a definite error - the item exists but is private
                        self.errors.push(ResolveError::UndefinedVar {
                            name: format!("{}::{} (private)", path, name),
                            span: miette::SourceSpan::from(0..1),
                        });
                    }
                    super::module_tree::ImportError::Circular { path } => {
                        self.errors.push(ResolveError::UndefinedVar {
                            name: format!("circular import: {}", path),
                            span: miette::SourceSpan::from(0..1),
                        });
                    }
                    super::module_tree::ImportError::Unresolved { name, path } => {
                        // Item not found in module - report as error
                        self.errors.push(ResolveError::UndefinedVar {
                            name: format!("{}::{} (not found)", path, name),
                            span: miette::SourceSpan::from(0..1),
                        });
                    }
                    super::module_tree::ImportError::ModuleNotFound { path } => {
                        // For now, defer module-not-found errors since they might
                        // be external modules loaded later
                        let _ = path;
                    }
                }
            }
        }

        // Build mapping from resolved import NodeIds to DefIds
        // This allows us to look up imported names during the second pass
        self.build_import_mappings();

        // Second pass: resolve bodies
        for item in &ast.items {
            self.resolve_item(item);
        }

        // Verify unresolved imports - retry resolution now that all modules are loaded
        self.verify_unresolved_imports();

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
            module_tree: self.module_tree,
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
        // Create a module ID from the name (for symbols)
        let parent_path = self.symbols.current_module().path.clone();
        let mut module_path = parent_path;
        module_path.push(m.name.clone());
        let module_id = ModuleId::new(module_path);

        // Create tree module ID and enter the module in the tree
        let tree_module_id = self.current_tree_module.join(&m.name);
        let parent_tree_id = self.current_tree_module.clone();

        // Create module in tree
        {
            let tree_module = self.module_tree.get_or_create(tree_module_id.clone());
            tree_module.visibility = m.visibility;
        }

        // Register as child of parent in tree
        if let Some(parent) = self.module_tree.get_mut(&parent_tree_id) {
            parent.add_child(m.name.clone(), tree_module_id.clone());
        }

        // Add module as item in parent
        if let Some(parent) = self.module_tree.get_mut(&parent_tree_id) {
            parent.add_item(ModuleItem {
                name: m.name.clone(),
                kind: ItemKind::Module(tree_module_id.clone()),
                visibility: m.visibility,
                node_id: m.id,
            });
        }

        // Enter the module in symbols
        self.symbols.enter_module(module_id.clone());

        // Push a new scope for the module so items defined in this module
        // are isolated from the parent scope
        self.symbols.push_scope(ScopeKind::Module, None);

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

        // Save current tree module and enter new one
        let saved_tree_module = self.current_tree_module.clone();
        self.current_tree_module = tree_module_id;

        // Recursively collect items if inline module
        if let Some(ref items) = m.items {
            for item in items {
                self.collect_item(item);
            }
        }

        // Exit module scope
        self.symbols.pop_scope();

        // Exit module
        self.current_tree_module = saved_tree_module;
        self.symbols.exit_module();
    }

    fn collect_import(&mut self, i: &ImportDef) {
        // Process import into the current module's symbol table
        let path: Vec<String> = i.path.segments.clone();
        if let Err(e) = self
            .symbols
            .process_import(&path, i.items.as_deref(), i.is_reexport)
        {
            // Track unresolved imports for verification after all modules loaded
            // This handles cases where imports reference modules loaded later
            self.unresolved_imports.push(PendingImport {
                path: path.clone(),
                reason: e,
                span: i.span,
            });
        }

        // Also record import in module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            match &i.items {
                Some(items) => {
                    for item in items {
                        let local_name = item.alias.clone().unwrap_or_else(|| item.name.clone());
                        module.add_import(ImportEntry {
                            local_name,
                            source_path: path.clone(),
                            original_name: item.name.clone(),
                            is_glob: item.is_glob,
                            is_reexport: i.is_reexport,
                            resolved: None,
                        });
                    }
                }
                None => {
                    // Import entire module
                    let module_name = path.last().cloned().unwrap_or_default();
                    module.add_import(ImportEntry {
                        local_name: module_name.clone(),
                        source_path: path.clone(),
                        original_name: module_name,
                        is_glob: false,
                        is_reexport: i.is_reexport,
                        resolved: None,
                    });
                }
            }
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

        // Add to module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            module.add_item(ModuleItem {
                name: f.name.clone(),
                kind: ItemKind::Function,
                visibility: f.visibility,
                node_id: f.id,
            });
        }
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

        // Add to module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            module.add_item(ModuleItem {
                name: s.name.clone(),
                kind: ItemKind::Struct,
                visibility: s.visibility,
                node_id: s.id,
            });
        }
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

        // Add to module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            module.add_item(ModuleItem {
                name: e.name.clone(),
                kind: ItemKind::Enum,
                visibility: e.visibility,
                node_id: e.id,
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

        // Add to module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            module.add_item(ModuleItem {
                name: t.name.clone(),
                kind: ItemKind::TypeAlias,
                visibility: t.visibility,
                node_id: t.id,
            });
        }
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

        // Add to module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            module.add_item(ModuleItem {
                name: e.name.clone(),
                kind: ItemKind::Effect,
                visibility: e.visibility,
                node_id: e.id,
            });
        }
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

        // Add to module tree
        if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
            module.add_item(ModuleItem {
                name: t.name.clone(),
                kind: ItemKind::Trait,
                visibility: t.visibility,
                node_id: t.id,
            });
        }
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

            // Add to module tree
            if let Some(module) = self.module_tree.get_mut(&self.current_tree_module) {
                module.add_item(ModuleItem {
                    name: name.clone(),
                    kind: ItemKind::Const,
                    visibility: g.visibility,
                    node_id: g.id,
                });
            }
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
        // Enter the module (symbols)
        let parent_path = self.symbols.current_module().path.clone();
        let mut module_path = parent_path;
        module_path.push(m.name.clone());
        let module_id = ModuleId::new(module_path);
        self.symbols.enter_module(module_id);

        // Enter the module (tree)
        let tree_module_id = self.current_tree_module.join(&m.name);
        let saved_tree_module = self.current_tree_module.clone();
        self.current_tree_module = tree_module_id;

        // Resolve items if inline module
        if let Some(ref items) = m.items {
            for item in items {
                self.resolve_item(item);
            }
        }

        // Exit module
        self.current_tree_module = saved_tree_module;
        self.symbols.exit_module();
    }

    fn resolve_function(&mut self, f: &FnDef) {
        let fn_def_id = self.symbols.def_for_node(f.id);
        self.symbols.push_scope(ScopeKind::Function, fn_def_id);

        // Resolve generic parameters
        for param in &f.generics.params {
            match param {
                GenericParam::Type { name, bounds, .. } => {
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
                GenericParam::Effect { name } => {
                    // Register effect parameter for row polymorphism
                    // Effect parameters are registered in the type namespace since
                    // effect references are resolved via resolve_path_as_type
                    let def_id = self.symbols.fresh_def_id();
                    let _ = self.symbols.define_type(name.clone(), def_id);
                    self.symbols.insert(Symbol {
                        def_id,
                        name: name.clone(),
                        kind: DefKind::EffectParam,
                        node_id: NodeId(0),
                        span: Span::default(),
                        parent: fn_def_id,
                    });
                }
                GenericParam::Const { .. } => {
                    // Const generics are handled separately
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
            TypeExpr::Forall { vars, inner } => {
                // For forall types, resolve the bounds and the inner type
                for v in vars {
                    for bound in &v.bounds {
                        self.resolve_path_as_type(bound);
                    }
                }
                self.resolve_type_expr(inner);
            }
        }
    }

    fn resolve_path_as_type(&mut self, path: &Path) {
        if path.is_simple() {
            let name = path.name().unwrap();
            // Use the new lookup that checks imports from ModuleTree
            if self.lookup_type_with_imports(name).is_none() {
                // Find similar type names for suggestion
                let suggestion = self.find_similar_type(name);
                self.errors.push(ResolveError::UndefinedType {
                    name: name.to_string(),
                    suggestion,
                    span: SourceSpan::from(0..1),
                });
            }
        } else {
            // Multi-segment type path (e.g., foo::Bar, std::collections::HashMap)
            if self.resolve_type_path_segments(&path.segments).is_none() {
                let suggestion = self.find_similar_type(path.segments.last().unwrap_or(&String::new()));
                self.errors.push(ResolveError::UndefinedType {
                    name: path.segments.join("::"),
                    suggestion,
                    span: SourceSpan::from(0..1),
                });
            }
        }
    }

    /// Find a similar type name using Levenshtein distance
    fn find_similar_type(&self, name: &str) -> Option<String> {
        let mut type_names = self.symbols.all_type_names();

        // Also include imported type names from the current module
        if let Some(module) = self.module_tree.get(&self.current_tree_module) {
            for import in &module.imports {
                if import.resolved.is_some() && !type_names.contains(&import.local_name) {
                    type_names.push(import.local_name.clone());
                }
            }
        }

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
                    // Use the new lookup that checks imports from ModuleTree
                    if let Some(def_id) = self.lookup_with_imports(name) {
                        self.symbols.record_ref(*id, def_id);
                    } else {
                        self.errors.push(ResolveError::UndefinedVar {
                            name: name.to_string(),
                            span: SourceSpan::from(0..1),
                        });
                    }
                } else {
                    // Multi-segment path (e.g., foo::Bar, std::collections::HashMap)
                    if let Some(def_id) = self.resolve_path_segments(&path.segments) {
                        self.symbols.record_ref(*id, def_id);
                    } else {
                        self.errors.push(ResolveError::UndefinedVar {
                            name: path.segments.join("::"),
                            span: SourceSpan::from(0..1),
                        });
                    }
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to parse and resolve source code
    fn resolve_source(source: &str) -> Result<ResolvedAst> {
        let tokens = crate::lexer::lex(source).expect("lexing failed");
        let ast = crate::parser::parse(&tokens, source).expect("parsing failed");
        resolve(ast)
    }

    /// Helper to check if resolution succeeds
    fn resolves_ok(source: &str) -> bool {
        resolve_source(source).is_ok()
    }

    /// Helper to check if resolution fails with a specific error pattern
    fn resolution_fails_with(source: &str, pattern: &str) -> bool {
        match resolve_source(source) {
            Ok(_) => false,
            Err(e) => e.to_string().contains(pattern),
        }
    }

    // ==================== Basic Import Tests ====================

    #[test]
    fn test_simple_import_from_module() {
        // Define a module with a public function, then import and use it
        // Note: Sounio uses 'module' keyword, not 'mod'
        let source = r#"
            module foo {
                pub fn bar() -> i32 { 42 }
            }
            use foo::bar;
            fn main() -> i32 {
                bar()
            }
        "#;
        assert!(resolves_ok(source), "Simple import should resolve");
    }

    #[test]
    fn test_import_struct() {
        let source = r#"
            module types {
                pub struct Point {
                    x: i32,
                    y: i32,
                }
            }
            use types::Point;
            fn make_point() -> Point {
                Point { x: 0, y: 0 }
            }
        "#;
        assert!(resolves_ok(source), "Import struct should resolve");
    }

    #[test]
    fn test_import_with_alias() {
        let source = r#"
            module math {
                pub fn calculate() -> i32 { 1 }
            }
            use math::calculate as calc;
            fn main() -> i32 {
                calc()
            }
        "#;
        assert!(resolves_ok(source), "Import with alias should resolve");
    }

    #[test]
    fn test_glob_import() {
        let source = r#"
            module utils {
                pub fn helper1() -> i32 { 1 }
                pub fn helper2() -> i32 { 2 }
            }
            use utils::*;
            fn main() -> i32 {
                helper1() + helper2()
            }
        "#;
        assert!(resolves_ok(source), "Glob import should resolve all public items");
    }

    #[test]
    fn test_selective_import() {
        let source = r#"
            module items {
                pub fn a() -> i32 { 1 }
                pub fn b() -> i32 { 2 }
                pub fn c() -> i32 { 3 }
            }
            use items::{a, c};
            fn main() -> i32 {
                a() + c()
            }
        "#;
        assert!(resolves_ok(source), "Selective import should resolve");
    }

    // ==================== Visibility Tests ====================

    #[test]
    fn test_private_item_not_visible() {
        let source = r#"
            module foo {
                fn private_fn() -> i32 { 42 }
            }
            use foo::private_fn;
            fn main() -> i32 {
                private_fn()
            }
        "#;
        // This should fail because private_fn is not public
        assert!(resolution_fails_with(source, "private") || resolution_fails_with(source, "not found"),
            "Private item should not be visible from outside");
    }

    #[test]
    fn test_glob_only_imports_public() {
        let source = r#"
            module foo {
                pub fn public_fn() -> i32 { 1 }
                fn private_fn() -> i32 { 2 }
            }
            use foo::*;
            fn main() -> i32 {
                public_fn()
            }
        "#;
        assert!(resolves_ok(source), "Glob should import public items");

        let source_with_private = r#"
            module foo {
                pub fn public_fn() -> i32 { 1 }
                fn private_fn() -> i32 { 2 }
            }
            use foo::*;
            fn main() -> i32 {
                private_fn()
            }
        "#;
        assert!(resolution_fails_with(source_with_private, "private_fn"),
            "Glob should not import private items");
    }

    #[test]
    fn test_qualified_path_private_item() {
        let source = r#"
            module foo {
                fn private_fn() -> i32 { 42 }
                pub fn public_fn() -> i32 { 1 }
            }
            fn main() -> i32 {
                foo::private_fn()
            }
        "#;
        assert!(resolution_fails_with(source, "private") || resolution_fails_with(source, "PrivateItem"),
            "Qualified path to private item should fail");

        // But public items should work
        let source_public = r#"
            module foo {
                fn private_fn() -> i32 { 42 }
                pub fn public_fn() -> i32 { 1 }
            }
            fn main() -> i32 {
                foo::public_fn()
            }
        "#;
        assert!(resolves_ok(source_public), "Qualified path to public item should work");
    }

    // ==================== Re-export Tests ====================

    #[test]
    fn test_reexport() {
        let source = r#"
            module internal {
                pub fn secret() -> i32 { 42 }
            }
            module facade {
                pub use internal::secret;
            }
            use facade::secret;
            fn main() -> i32 {
                secret()
            }
        "#;
        // Note: This tests transitive imports through re-exports
        assert!(resolves_ok(source), "Re-export should make item visible");
    }

    // ==================== Path Resolution Tests ====================

    #[test]
    fn test_qualified_path() {
        let source = r#"
            module outer {
                pub module inner {
                    pub fn deep() -> i32 { 42 }
                }
            }
            fn main() -> i32 {
                outer::inner::deep()
            }
        "#;
        assert!(resolves_ok(source), "Qualified path should resolve");
    }

    #[test]
    fn test_qualified_type_path() {
        let source = r#"
            module types {
                pub struct Widget {
                    value: i32,
                }
            }
            fn create() -> types::Widget {
                types::Widget { value: 0 }
            }
        "#;
        assert!(resolves_ok(source), "Qualified type path should resolve");
    }

    // ==================== Error Cases ====================

    #[test]
    fn test_undefined_import() {
        let source = r#"
            module foo {
                pub fn bar() -> i32 { 1 }
            }
            use foo::nonexistent;
            fn main() -> i32 { 0 }
        "#;
        assert!(resolution_fails_with(source, "nonexistent") || resolution_fails_with(source, "not found"),
            "Import of nonexistent item should fail");
    }

    #[test]
    fn test_import_from_nonexistent_module() {
        let source = r#"
            use nonexistent_module::something;
            fn main() -> i32 { 0 }
        "#;
        // Module not found errors might be deferred for external modules
        // but we should at least not crash
        let _ = resolve_source(source);
    }

    // ==================== Nested Module Tests ====================

    #[test]
    fn test_nested_module_import() {
        let source = r#"
            module a {
                pub module b {
                    pub module c {
                        pub fn deep_fn() -> i32 { 42 }
                    }
                }
            }
            use a::b::c::deep_fn;
            fn main() -> i32 {
                deep_fn()
            }
        "#;
        assert!(resolves_ok(source), "Deeply nested import should resolve");
    }

    #[test]
    fn test_sibling_module_import() {
        let source = r#"
            module alpha {
                pub fn from_alpha() -> i32 { 1 }
            }
            module beta {
                use alpha::from_alpha;
                pub fn from_beta() -> i32 {
                    from_alpha()
                }
            }
            fn main() -> i32 {
                beta::from_beta()
            }
        "#;
        assert!(resolves_ok(source), "Sibling module import should resolve");
    }

    // ==================== Type Import Tests ====================

    #[test]
    fn test_import_enum() {
        let source = r#"
            module colors {
                pub enum Color {
                    Red,
                    Green,
                    Blue,
                }
            }
            use colors::Color;
            fn get_color() -> Color {
                Color::Red
            }
        "#;
        assert!(resolves_ok(source), "Import enum should resolve");
    }

    #[test]
    fn test_import_type_alias() {
        let source = r#"
            module aliases {
                pub type Id = i32;
            }
            use aliases::Id;
            fn get_id() -> Id {
                42
            }
        "#;
        assert!(resolves_ok(source), "Import type alias should resolve");
    }

    // ==================== Complex Scenarios ====================

    #[test]
    fn test_multiple_imports_from_same_module() {
        let source = r#"
            module utils {
                pub fn func1() -> i32 { 1 }
                pub fn func2() -> i32 { 2 }
                pub struct Data { value: i32 }
            }
            use utils::func1;
            use utils::func2;
            use utils::Data;
            fn main() -> i32 {
                let d = Data { value: func1() };
                func2()
            }
        "#;
        assert!(resolves_ok(source), "Multiple imports from same module should resolve");
    }

    #[test]
    fn test_import_does_not_shadow_local() {
        let source = r#"
            module external {
                pub fn helper() -> i32 { 1 }
            }
            fn helper() -> i32 { 2 }
            use external::*;
            fn main() -> i32 {
                helper()
            }
        "#;
        // Local definitions should take precedence over glob imports
        assert!(resolves_ok(source), "Local should not be shadowed by glob import");
    }
}
