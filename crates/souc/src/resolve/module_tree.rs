//! Module tree construction and management
//!
//! This module provides data structures for representing the module hierarchy
//! and tracking items within modules for name resolution.
//!
//! ## Import Resolution
//!
//! Import resolution uses a multi-pass algorithm to handle transitive imports:
//! 1. Collect all imports in the first pass during tree construction
//! 2. Iteratively resolve imports until no more progress can be made
//! 3. Report any remaining unresolved imports as errors
//!
//! This approach gracefully handles:
//! - Forward references (importing something not yet defined)
//! - Transitive imports (A imports B which re-exports C)
//! - Circular imports (detects and reports cycles)

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::ast::Visibility;
use crate::common::NodeId;
use std::collections::HashMap;

/// Import resolution error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImportError {
    /// Import could not be resolved - item not found
    Unresolved {
        /// Name of the item being imported
        name: String,
        /// Full path being imported from
        path: String,
    },
    /// Import target exists but is not visible
    NotVisible {
        /// Name of the item being imported
        name: String,
        /// Full path being imported from
        path: String,
    },
    /// Circular import detected
    Circular {
        /// Path involved in the cycle
        path: String,
    },
    /// Module not found
    ModuleNotFound {
        /// Path to the module that was not found
        path: String,
    },
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Unresolved { name, path } => {
                write!(f, "unresolved import: `{}` not found in `{}`", name, path)
            }
            ImportError::NotVisible { name, path } => {
                write!(f, "import error: `{}` in `{}` is private", name, path)
            }
            ImportError::Circular { path } => {
                write!(f, "circular import detected involving `{}`", path)
            }
            ImportError::ModuleNotFound { path } => {
                write!(f, "module not found: `{}`", path)
            }
        }
    }
}

impl std::error::Error for ImportError {}

/// Unique module identifier
///
/// Represents a path through the module hierarchy (e.g., `["std", "collections"]`).
/// The root module has an empty path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModuleId(pub Vec<String>);

impl ModuleId {
    /// Create the root module ID (empty path)
    pub fn root() -> Self {
        Self(Vec::new())
    }

    /// Create a new module ID by joining a name to this path
    pub fn join(&self, name: &str) -> Self {
        let mut path = self.0.clone();
        path.push(name.to_string());
        Self(path)
    }

    /// Get the parent module ID, or None if this is the root
    pub fn parent(&self) -> Option<Self> {
        if self.0.is_empty() {
            None
        } else {
            let mut path = self.0.clone();
            path.pop();
            Some(Self(path))
        }
    }

    /// Get the module name (last segment), or None for root
    pub fn name(&self) -> Option<&str> {
        self.0.last().map(|s| s.as_str())
    }

    /// Check if this is the root module
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the path segments
    pub fn segments(&self) -> &[String] {
        &self.0
    }
}

impl std::fmt::Display for ModuleId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0.is_empty() {
            write!(f, "<root>")
        } else {
            write!(f, "{}", self.0.join("::"))
        }
    }
}

impl From<Vec<String>> for ModuleId {
    fn from(path: Vec<String>) -> Self {
        Self(path)
    }
}

impl From<&[String]> for ModuleId {
    fn from(path: &[String]) -> Self {
        Self(path.to_vec())
    }
}

/// Module in the tree
#[derive(Debug, Clone)]
pub struct Module {
    /// Unique identifier for this module
    pub id: ModuleId,
    /// Module name (last segment of path)
    pub name: String,
    /// Visibility of this module
    pub visibility: Visibility,
    /// Parent module ID (None for root)
    pub parent: Option<ModuleId>,
    /// Child modules: name -> ModuleId
    pub children: HashMap<String, ModuleId>,
    /// Items defined in this module
    pub items: Vec<ModuleItem>,
    /// Import entries
    pub imports: Vec<ImportEntry>,
}

/// Item defined in a module
#[derive(Debug, Clone)]
pub struct ModuleItem {
    /// Item name
    pub name: String,
    /// Kind of item
    pub kind: ItemKind,
    /// Visibility
    pub visibility: Visibility,
    /// AST node ID for this item
    pub node_id: NodeId,
}

/// Kind of item in a module
#[derive(Debug, Clone)]
pub enum ItemKind {
    /// Function definition
    Function,
    /// Struct definition
    Struct,
    /// Enum definition
    Enum,
    /// Trait definition
    Trait,
    /// Type alias
    TypeAlias,
    /// Effect definition
    Effect,
    /// Handler definition
    Handler,
    /// Constant or static
    Const,
    /// Nested module
    Module(ModuleId),
}

/// Import entry tracking what is imported into a module
#[derive(Debug, Clone)]
pub struct ImportEntry {
    /// Local name (what this import is called in the current module)
    pub local_name: String,
    /// Source path (the module path being imported from)
    pub source_path: Vec<String>,
    /// Original name in the source module
    pub original_name: String,
    /// Whether this is a glob import (`use foo::*`)
    pub is_glob: bool,
    /// Whether this is a re-export (`pub use`)
    pub is_reexport: bool,
    /// Whether this imports a module itself (vs an item from a module)
    /// For `use foo::bar;` where bar is a module, this is true.
    /// For `use foo::{item}` or `use foo::item` where item is a function/type, this is false.
    pub is_module_import: bool,
    /// Resolved node ID (set during resolution)
    pub resolved: Option<NodeId>,
}

/// The complete module tree
#[derive(Debug, Clone)]
pub struct ModuleTree {
    /// All modules indexed by their ID
    modules: HashMap<ModuleId, Module>,
    /// The root module ID
    root: ModuleId,
    /// Mapping of ModuleId to allocated NodeId
    module_node_ids: HashMap<ModuleId, NodeId>,
    /// Counter for allocating new module NodeIds (start from 1000000 to avoid collisions)
    next_module_node_id: u32,
}

impl ModuleTree {
    /// Create a new module tree with an empty root module
    pub fn new() -> Self {
        let root = ModuleId::root();
        let mut modules = HashMap::new();
        let mut module_node_ids = HashMap::new();

        // Allocate the first module NodeId for root
        module_node_ids.insert(root.clone(), NodeId(1000000));

        modules.insert(
            root.clone(),
            Module {
                id: root.clone(),
                name: String::new(),
                visibility: Visibility::Public,
                parent: None,
                children: HashMap::new(),
                items: Vec::new(),
                imports: Vec::new(),
            },
        );
        Self {
            modules,
            root,
            module_node_ids,
            next_module_node_id: 1000001,
        }
    }

    /// Get the root module ID
    pub fn root(&self) -> &ModuleId {
        &self.root
    }

    /// Get a module by ID
    pub fn get(&self, id: &ModuleId) -> Option<&Module> {
        self.modules.get(id)
    }

    /// Get a mutable reference to a module by ID
    pub fn get_mut(&mut self, id: &ModuleId) -> Option<&mut Module> {
        self.modules.get_mut(id)
    }

    /// Get or allocate a NodeId for a module
    pub fn get_module_node_id(&mut self, module_id: &ModuleId) -> NodeId {
        if let Some(&node_id) = self.module_node_ids.get(module_id) {
            node_id
        } else {
            let node_id = NodeId(self.next_module_node_id);
            self.next_module_node_id += 1;
            self.module_node_ids.insert(module_id.clone(), node_id);
            node_id
        }
    }

    /// Insert a module into the tree
    pub fn insert(&mut self, module: Module) {
        self.modules.insert(module.id.clone(), module);
    }

    /// Iterate over all module IDs
    pub fn all_module_ids(&self) -> impl Iterator<Item = &ModuleId> {
        self.modules.keys()
    }

    /// Iterate over all modules
    pub fn all_modules(&self) -> impl Iterator<Item = &Module> {
        self.modules.values()
    }

    /// Check if a module exists
    pub fn contains(&self, id: &ModuleId) -> bool {
        self.modules.contains_key(id)
    }

    /// Get or create a module at the given path
    pub fn get_or_create(&mut self, id: ModuleId) -> &mut Module {
        if !self.modules.contains_key(&id) {
            let parent = id.parent();
            let name = id.name().unwrap_or("").to_string();

            // Ensure parent exists
            if let Some(ref parent_id) = parent {
                self.get_or_create(parent_id.clone());
                // Register this as a child of parent
                if let Some(parent_mod) = self.modules.get_mut(parent_id) {
                    parent_mod.children.insert(name.clone(), id.clone());
                }
            }

            self.modules.insert(
                id.clone(),
                Module {
                    id: id.clone(),
                    name,
                    visibility: Visibility::Private,
                    parent,
                    children: HashMap::new(),
                    items: Vec::new(),
                    imports: Vec::new(),
                },
            );
        }
        self.modules.get_mut(&id).unwrap()
    }

    /// Resolve all imports in the module tree using a multi-pass algorithm
    ///
    /// This iteratively resolves imports until no more progress can be made.
    /// The multi-pass approach handles:
    /// - Forward references
    /// - Transitive imports (A imports B's re-export of C)
    /// - Circular dependencies (detected and reported)
    ///
    /// Returns Ok(()) if all imports resolved, Err with list of unresolved imports otherwise.
    pub fn resolve_imports(&mut self) -> Result<(), Vec<ImportError>> {
        let mut errors = Vec::new();
        let mut changed = true;
        let mut passes = 0;
        const MAX_PASSES: usize = 100;

        while changed && passes < MAX_PASSES {
            changed = false;
            passes += 1;

            // Collect all module IDs to iterate over
            let module_ids: Vec<_> = self.all_module_ids().cloned().collect();

            for module_id in module_ids {
                // Get unresolved imports for this module
                let unresolved: Vec<_> = {
                    let module = match self.get(&module_id) {
                        Some(m) => m,
                        None => continue,
                    };
                    module
                        .imports
                        .iter()
                        .enumerate()
                        .filter(|(_, imp)| imp.resolved.is_none() && !imp.is_glob)
                        .map(|(i, imp)| (i, imp.clone()))
                        .collect()
                };

                for (idx, import) in unresolved {
                    // Skip module imports - they don't need item resolution
                    if import.is_module_import {
                        // For module imports, just verify the module exists
                        let target_id = ModuleId(import.source_path.clone());
                        if self.contains(&target_id) {
                            // Mark as resolved (no specific item, just the module)
                            if let Some(module) = self.get_mut(&module_id) {
                                module.imports[idx].resolved = Some(NodeId(0)); // Sentinel for "module resolved"
                            }
                            changed = true;
                        }
                        continue;
                    }

                    match self.try_resolve_import(&module_id, &import) {
                        ResolveResult::Resolved(node_id) => {
                            if let Some(module) = self.get_mut(&module_id) {
                                module.imports[idx].resolved = Some(node_id);
                                changed = true;
                            }
                        }
                        ResolveResult::NotVisible => {
                            // Record visibility error immediately
                            errors.push(ImportError::NotVisible {
                                name: import.original_name.clone(),
                                path: import.source_path.join("::"),
                            });
                            // Mark as "resolved" with a sentinel to avoid re-processing
                            if let Some(module) = self.get_mut(&module_id) {
                                module.imports[idx].resolved = Some(NodeId(u32::MAX));
                            }
                        }
                        ResolveResult::NotFound | ResolveResult::ModuleNotFound => {
                            // Will be checked again in next pass or reported at end
                        }
                    }
                }

                // Handle glob imports separately
                self.resolve_glob_imports_for_module(&module_id);
            }
        }

        // Collect remaining unresolved imports as errors
        for module_id in self.all_module_ids() {
            let module = match self.get(module_id) {
                Some(m) => m,
                None => continue,
            };
            for import in &module.imports {
                if import.resolved.is_none() && !import.is_module_import {
                    // Check if it's a module not found vs item not found
                    let target_id = ModuleId(import.source_path.clone());
                    if !self.contains(&target_id) {
                        errors.push(ImportError::ModuleNotFound {
                            path: import.source_path.join("::"),
                        });
                    } else if !import.is_glob {
                        // For non-glob imports, report specific item not found
                        errors.push(ImportError::Unresolved {
                            name: import.original_name.clone(),
                            path: import.source_path.join("::"),
                        });
                    }
                    // For glob imports where module exists but has no public items,
                    // this is not an error - the glob just imports nothing
                }
            }
        }

        // Check for circular imports (if we hit max passes without resolving)
        if passes >= MAX_PASSES && changed {
            errors.push(ImportError::Circular {
                path: "unknown".to_string(),
            });
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Try to resolve a single import
    fn try_resolve_import(&mut self, from: &ModuleId, import: &ImportEntry) -> ResolveResult {
        // Build target module path
        let target_id = ModuleId(import.source_path.clone());

        // Get target module
        let target = match self.get(&target_id) {
            Some(m) => m,
            None => return ResolveResult::ModuleNotFound,
        };

        // Check if it's accessing itself
        let from_same = from == &target_id;

        // First try direct lookup
        if let Some(item) = target.items.iter().find(|i| i.name == import.original_name) {
            // Check visibility (Sounio has no `pub`, so all items are accessible)
            if target.is_visible(item, from) {
                return ResolveResult::Resolved(item.node_id);
            } else {
                return ResolveResult::NotVisible;
            }
        }

        // Try looking up through re-exports in the target module
        for target_import in &target.imports {
            if target_import.is_reexport
                && target_import.local_name == import.original_name
                && target_import.resolved.is_some()
            {
                return ResolveResult::Resolved(target_import.resolved.unwrap());
            }
        }

        // Check child modules (e.g., `use foo::bar` where bar is a module)
        if let Some(child_id) = target.children.get(&import.original_name) {
            // Clone child_id before borrowing self again
            let child_id_cloned = child_id.clone();
            if let Some(child_mod) = self.get(&child_id_cloned) {
                // Find the module item in parent that represents this child
                if let Some(item) = target.items.iter().find(|i| {
                    i.name == import.original_name && matches!(i.kind, ItemKind::Module(_))
                }) {
                    if target.is_visible(item, from) {
                        return ResolveResult::Resolved(item.node_id);
                    } else {
                        return ResolveResult::NotVisible;
                    }
                }
                // If module exists but no item entry, it's implicitly public
                // Return the allocated NodeId for this module
                // Clone the id before the mutable borrow
                let mod_id = child_mod.id.clone();
                let node_id = self.get_module_node_id(&mod_id);
                return ResolveResult::Resolved(node_id);
            }
        }

        ResolveResult::NotFound
    }

    /// Resolve glob imports for a specific module
    fn resolve_glob_imports_for_module(&mut self, module_id: &ModuleId) {
        // Collect glob imports that need processing
        let glob_imports: Vec<ImportEntry> = {
            let module = match self.get(module_id) {
                Some(m) => m,
                None => return,
            };
            module
                .imports
                .iter()
                .filter(|imp| imp.is_glob && imp.resolved.is_none())
                .cloned()
                .collect()
        };

        for glob_import in glob_imports {
            let target_id = ModuleId(glob_import.source_path.clone());

            // Collect items to import
            let items_to_import: Vec<(String, NodeId, Visibility)> = {
                let target = match self.get(&target_id) {
                    Some(m) => m,
                    None => continue,
                };

                // Get all public items from target
                let mut items: Vec<_> = target
                    .public_items()
                    .map(|item| (item.name.clone(), item.node_id, item.visibility))
                    .collect();

                // Also include resolved re-exports
                for imp in &target.imports {
                    if imp.is_reexport && imp.resolved.is_some() {
                        items.push((
                            imp.local_name.clone(),
                            imp.resolved.unwrap(),
                            Visibility::Public,
                        ));
                    }
                }

                items
            };

            // Add imported items to the module
            if let Some(module) = self.get_mut(module_id) {
                // Mark glob as resolved
                for imp in &mut module.imports {
                    if imp.is_glob && imp.source_path == glob_import.source_path {
                        imp.resolved = Some(NodeId(0)); // Sentinel for "resolved glob"
                        break;
                    }
                }

                // Add synthetic import entries for each imported item
                for (name, node_id, _visibility) in items_to_import {
                    // Don't add if already exists
                    let already_exists = module
                        .imports
                        .iter()
                        .any(|i| i.local_name == name && !i.is_glob)
                        || module.items.iter().any(|i| i.name == name);

                    if !already_exists {
                        module.imports.push(ImportEntry {
                            local_name: name.clone(),
                            source_path: glob_import.source_path.clone(),
                            original_name: name,
                            is_glob: false,
                            is_reexport: glob_import.is_reexport,
                            is_module_import: false,
                            resolved: Some(node_id),
                        });
                    }
                }
            }
        }
    }

    /// Resolve a path to a module ID
    ///
    /// Given a path like ["std", "collections"], returns the ModuleId if the module exists.
    pub fn resolve_module_path(&self, path: &[String]) -> Option<&ModuleId> {
        let target_id = ModuleId(path.to_vec());
        if self.contains(&target_id) {
            // Return a reference to the key in the map
            self.modules.keys().find(|id| **id == target_id)
        } else {
            None
        }
    }

    /// Get all items visible in a module (including imports)
    pub fn visible_items_in(&self, module_id: &ModuleId) -> Vec<(String, NodeId)> {
        let module = match self.get(module_id) {
            Some(m) => m,
            None => return Vec::new(),
        };

        let mut items = Vec::new();

        // Direct items
        for item in &module.items {
            items.push((item.name.clone(), item.node_id));
        }

        // Resolved imports
        for imp in &module.imports {
            if let Some(node_id) = imp.resolved {
                if node_id.0 != u32::MAX {
                    // Not an error sentinel
                    items.push((imp.local_name.clone(), node_id));
                }
            }
        }

        items
    }
}

/// Result of attempting to resolve an import
#[derive(Debug, Clone, PartialEq, Eq)]
enum ResolveResult {
    /// Successfully resolved to a NodeId
    Resolved(NodeId),
    /// Item exists but is not visible
    NotVisible,
    /// Item not found in module
    NotFound,
    /// Module itself not found
    ModuleNotFound,
}

impl Default for ModuleTree {
    fn default() -> Self {
        Self::new()
    }
}

impl Module {
    /// Create a new module
    pub fn new(id: ModuleId, name: String, parent: Option<ModuleId>) -> Self {
        Self {
            id,
            name,
            visibility: Visibility::Private,
            parent,
            children: HashMap::new(),
            items: Vec::new(),
            imports: Vec::new(),
        }
    }

    /// Add a child module
    pub fn add_child(&mut self, name: String, child_id: ModuleId) {
        self.children.insert(name, child_id);
    }

    /// Add an item to this module
    pub fn add_item(&mut self, item: ModuleItem) {
        self.items.push(item);
    }

    /// Look up an item by name, respecting visibility
    ///
    /// If `from_same_module` is true, private items are visible.
    /// Otherwise, only public items are returned.
    pub fn lookup(&self, name: &str, from_same_module: bool) -> Option<&ModuleItem> {
        self.items.iter().find(|item| {
            item.name == name && (from_same_module || matches!(item.visibility, Visibility::Public))
        })
    }

    /// Get all public items
    pub fn public_items(&self) -> impl Iterator<Item = &ModuleItem> {
        self.items
            .iter()
            .filter(|item| matches!(item.visibility, Visibility::Public))
    }

    /// Add an import entry
    pub fn add_import(&mut self, import: ImportEntry) {
        self.imports.push(import);
    }

    /// Check if an item is visible from another module
    ///
    /// Visibility rules:
    /// - Public items are always visible
    /// - Private items are only visible within the same module or its descendants
    pub fn is_visible(&self, item: &ModuleItem, from: &ModuleId) -> bool {
        match item.visibility {
            Visibility::Public => true,
            Visibility::Private => {
                // Sounio has no `pub` keyword — all items default to Private but
                // are intended to be accessible across modules. Treat as visible.
                true
            }
        }
    }

    /// Look up an item by name with full visibility check from a specific module
    pub fn lookup_from(&self, name: &str, from: &ModuleId) -> Option<&ModuleItem> {
        self.items
            .iter()
            .find(|item| item.name == name && self.is_visible(item, from))
    }

    /// Get all items visible from another module
    pub fn visible_items_from(&self, from: &ModuleId) -> impl Iterator<Item = &ModuleItem> {
        let from_clone = from.clone();
        let self_id = self.id.clone();
        self.items.iter().filter(move |item| match item.visibility {
            Visibility::Public => true,
            Visibility::Private => {
                if from_clone.0.len() < self_id.0.len() {
                    false
                } else {
                    from_clone.0.starts_with(&self_id.0)
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_id_operations() {
        let root = ModuleId::root();
        assert!(root.is_root());
        assert_eq!(root.name(), None);
        assert_eq!(root.parent(), None);

        let std = root.join("std");
        assert!(!std.is_root());
        assert_eq!(std.name(), Some("std"));
        assert_eq!(std.parent(), Some(ModuleId::root()));

        let collections = std.join("collections");
        assert_eq!(collections.name(), Some("collections"));
        assert_eq!(collections.parent(), Some(std.clone()));
        assert_eq!(collections.to_string(), "std::collections");
    }

    #[test]
    fn test_module_tree_creation() {
        let mut tree = ModuleTree::new();

        // Root exists
        assert!(tree.get(tree.root()).is_some());

        // Create nested module
        let std_id = ModuleId::root().join("std");
        let std_mod = tree.get_or_create(std_id.clone());
        std_mod.visibility = Visibility::Public;

        let collections_id = std_id.join("collections");
        tree.get_or_create(collections_id.clone());

        // Verify structure
        let root = tree.get(&ModuleId::root()).unwrap();
        assert!(root.children.contains_key("std"));

        let std = tree.get(&std_id).unwrap();
        assert!(std.children.contains_key("collections"));
    }

    #[test]
    fn test_module_item_lookup() {
        let mut module = Module::new(
            ModuleId::root().join("test"),
            "test".to_string(),
            Some(ModuleId::root()),
        );

        // Add public item
        module.add_item(ModuleItem {
            name: "public_fn".to_string(),
            kind: ItemKind::Function,
            visibility: Visibility::Public,
            node_id: NodeId(1),
        });

        // Add private item
        module.add_item(ModuleItem {
            name: "private_fn".to_string(),
            kind: ItemKind::Function,
            visibility: Visibility::Private,
            node_id: NodeId(2),
        });

        // From same module: both visible
        assert!(module.lookup("public_fn", true).is_some());
        assert!(module.lookup("private_fn", true).is_some());

        // From different module: only public visible
        assert!(module.lookup("public_fn", false).is_some());
        assert!(module.lookup("private_fn", false).is_none());
    }

    // ==================== Import Resolution Tests ====================

    /// Helper to create a simple module tree for testing imports
    fn create_test_tree_with_modules() -> ModuleTree {
        let mut tree = ModuleTree::new();

        // Create module "foo" with public items
        let foo_id = ModuleId::root().join("foo");
        {
            let foo = tree.get_or_create(foo_id.clone());
            foo.visibility = Visibility::Public;
            foo.add_item(ModuleItem {
                name: "Bar".to_string(),
                kind: ItemKind::Struct,
                visibility: Visibility::Public,
                node_id: NodeId(100),
            });
            foo.add_item(ModuleItem {
                name: "Baz".to_string(),
                kind: ItemKind::Struct,
                visibility: Visibility::Public,
                node_id: NodeId(101),
            });
            foo.add_item(ModuleItem {
                name: "private_item".to_string(),
                kind: ItemKind::Function,
                visibility: Visibility::Private,
                node_id: NodeId(102),
            });
        }

        tree
    }

    #[test]
    fn test_simple_import() {
        // Test: use foo::Bar
        let mut tree = create_test_tree_with_modules();

        // Add import to root module
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "Bar".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "Bar".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        // Resolve imports
        let result = tree.resolve_imports();
        assert!(result.is_ok(), "Import resolution should succeed");

        // Check that import was resolved
        let root = tree.get(&ModuleId::root()).unwrap();
        let import = root.imports.iter().find(|i| i.local_name == "Bar").unwrap();
        assert_eq!(import.resolved, Some(NodeId(100)));
    }

    #[test]
    fn test_selective_imports() {
        // Test: use foo::{Bar, Baz}
        let mut tree = create_test_tree_with_modules();

        // Add selective imports to root module
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "Bar".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "Bar".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
            root.add_import(ImportEntry {
                local_name: "Baz".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "Baz".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_ok());

        let root = tree.get(&ModuleId::root()).unwrap();
        let bar_import = root.imports.iter().find(|i| i.local_name == "Bar").unwrap();
        let baz_import = root.imports.iter().find(|i| i.local_name == "Baz").unwrap();
        assert_eq!(bar_import.resolved, Some(NodeId(100)));
        assert_eq!(baz_import.resolved, Some(NodeId(101)));
    }

    #[test]
    fn test_glob_import() {
        // Test: use foo::*
        let mut tree = create_test_tree_with_modules();

        // Add glob import to root module
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "*".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "*".to_string(),
                is_glob: true,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_ok());

        // Check that public items were imported
        let root = tree.get(&ModuleId::root()).unwrap();

        // Should have Bar and Baz (public), but not private_item
        let bar_import = root.imports.iter().find(|i| i.local_name == "Bar");
        let baz_import = root.imports.iter().find(|i| i.local_name == "Baz");
        let private_import = root.imports.iter().find(|i| i.local_name == "private_item");

        assert!(bar_import.is_some(), "Bar should be imported via glob");
        assert!(baz_import.is_some(), "Baz should be imported via glob");
        assert!(
            private_import.is_none(),
            "private_item should not be imported"
        );

        assert_eq!(bar_import.unwrap().resolved, Some(NodeId(100)));
        assert_eq!(baz_import.unwrap().resolved, Some(NodeId(101)));
    }

    #[test]
    fn test_reexport() {
        // Test: pub use internal::Thing in foo, then use foo::Thing elsewhere
        let mut tree = ModuleTree::new();

        // Create internal module with Item
        let internal_id = ModuleId::root().join("internal");
        {
            let internal = tree.get_or_create(internal_id.clone());
            internal.visibility = Visibility::Private; // Internal module is private
            internal.add_item(ModuleItem {
                name: "Thing".to_string(),
                kind: ItemKind::Struct,
                visibility: Visibility::Public,
                node_id: NodeId(200),
            });
        }

        // Create foo module that re-exports Thing
        let foo_id = ModuleId::root().join("foo");
        {
            let foo = tree.get_or_create(foo_id.clone());
            foo.visibility = Visibility::Public;
            foo.add_import(ImportEntry {
                local_name: "Thing".to_string(),
                source_path: vec!["internal".to_string()],
                original_name: "Thing".to_string(),
                is_glob: false,
                is_reexport: true, // pub use
                is_module_import: false,
                resolved: None,
            });
        }

        // Root imports Thing from foo
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "Thing".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "Thing".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_ok(), "Re-export resolution should succeed");

        // Check that Thing was resolved in root
        let root = tree.get(&ModuleId::root()).unwrap();
        let thing_import = root
            .imports
            .iter()
            .find(|i| i.local_name == "Thing")
            .unwrap();
        assert_eq!(thing_import.resolved, Some(NodeId(200)));
    }

    #[test]
    fn test_visibility_error() {
        // Test: Sounio has no `pub` keyword, so importing a "private" item should succeed
        let mut tree = create_test_tree_with_modules();

        // Import a private item — should work because all items are accessible
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "private_item".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "private_item".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(
            result.is_ok(),
            "Sounio has no pub — private items should be importable"
        );

        // Verify the import resolved to the correct node
        let root = tree.get(&ModuleId::root()).unwrap();
        let import = root
            .imports
            .iter()
            .find(|i| i.local_name == "private_item")
            .unwrap();
        assert_eq!(import.resolved, Some(NodeId(102)));
    }

    #[test]
    fn test_unresolved_import() {
        // Test: importing a non-existent item
        let mut tree = create_test_tree_with_modules();

        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "NonExistent".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "NonExistent".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_err(), "Should fail for non-existent item");

        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(
                |e| matches!(e, ImportError::Unresolved { name, .. } if name == "NonExistent")
            )
        );
    }

    #[test]
    fn test_module_not_found() {
        // Test: importing from a non-existent module
        let mut tree = ModuleTree::new();

        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "Item".to_string(),
                source_path: vec!["nonexistent".to_string()],
                original_name: "Item".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_err());

        let errors = result.unwrap_err();
        assert!(
            errors.iter().any(
                |e| matches!(e, ImportError::ModuleNotFound { path } if path == "nonexistent")
            )
        );
    }

    #[test]
    fn test_transitive_imports() {
        // Test: a -> b -> c (transitive re-exports)
        let mut tree = ModuleTree::new();

        // Module c has the actual item
        let c_id = ModuleId::root().join("c");
        {
            let c = tree.get_or_create(c_id.clone());
            c.visibility = Visibility::Public;
            c.add_item(ModuleItem {
                name: "DeepItem".to_string(),
                kind: ItemKind::Struct,
                visibility: Visibility::Public,
                node_id: NodeId(300),
            });
        }

        // Module b re-exports from c
        let b_id = ModuleId::root().join("b");
        {
            let b = tree.get_or_create(b_id.clone());
            b.visibility = Visibility::Public;
            b.add_import(ImportEntry {
                local_name: "DeepItem".to_string(),
                source_path: vec!["c".to_string()],
                original_name: "DeepItem".to_string(),
                is_glob: false,
                is_reexport: true,
                is_module_import: false,
                resolved: None,
            });
        }

        // Module a re-exports from b
        let a_id = ModuleId::root().join("a");
        {
            let a = tree.get_or_create(a_id.clone());
            a.visibility = Visibility::Public;
            a.add_import(ImportEntry {
                local_name: "DeepItem".to_string(),
                source_path: vec!["b".to_string()],
                original_name: "DeepItem".to_string(),
                is_glob: false,
                is_reexport: true,
                is_module_import: false,
                resolved: None,
            });
        }

        // Root imports from a
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "DeepItem".to_string(),
                source_path: vec!["a".to_string()],
                original_name: "DeepItem".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_ok(), "Transitive imports should resolve");

        let root = tree.get(&ModuleId::root()).unwrap();
        let import = root
            .imports
            .iter()
            .find(|i| i.local_name == "DeepItem")
            .unwrap();
        assert_eq!(import.resolved, Some(NodeId(300)));
    }

    #[test]
    fn test_visibility_check_same_module() {
        let module = Module::new(
            ModuleId(vec!["foo".to_string()]),
            "foo".to_string(),
            Some(ModuleId::root()),
        );

        let public_item = ModuleItem {
            name: "public".to_string(),
            kind: ItemKind::Function,
            visibility: Visibility::Public,
            node_id: NodeId(1),
        };

        let private_item = ModuleItem {
            name: "private".to_string(),
            kind: ItemKind::Function,
            visibility: Visibility::Private,
            node_id: NodeId(2),
        };

        // From the same module - both visible
        let from_same = ModuleId(vec!["foo".to_string()]);
        assert!(module.is_visible(&public_item, &from_same));
        assert!(module.is_visible(&private_item, &from_same));

        // From a child module - both visible
        let from_child = ModuleId(vec!["foo".to_string(), "bar".to_string()]);
        assert!(module.is_visible(&public_item, &from_child));
        assert!(module.is_visible(&private_item, &from_child));

        // From a different module - Sounio has no `pub`, all items visible
        let from_other = ModuleId(vec!["other".to_string()]);
        assert!(module.is_visible(&public_item, &from_other));
        assert!(module.is_visible(&private_item, &from_other));

        // From root - Sounio has no `pub`, all items visible
        let from_root = ModuleId::root();
        assert!(module.is_visible(&public_item, &from_root));
        assert!(module.is_visible(&private_item, &from_root));
    }

    #[test]
    fn test_visible_items_in_module() {
        let mut tree = create_test_tree_with_modules();

        // Add an import to root
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "Bar".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "Bar".to_string(),
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        tree.resolve_imports().unwrap();

        // Check visible items in root
        let visible = tree.visible_items_in(&ModuleId::root());
        assert!(visible.iter().any(|(name, _)| name == "Bar"));
    }

    #[test]
    fn test_aliased_import() {
        // Test: use foo::Bar as MyBar
        let mut tree = create_test_tree_with_modules();

        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "MyBar".to_string(), // alias
                source_path: vec!["foo".to_string()],
                original_name: "Bar".to_string(), // original name
                is_glob: false,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        let result = tree.resolve_imports();
        assert!(result.is_ok());

        let root = tree.get(&ModuleId::root()).unwrap();
        let import = root
            .imports
            .iter()
            .find(|i| i.local_name == "MyBar")
            .unwrap();
        assert_eq!(import.resolved, Some(NodeId(100)));
        assert_eq!(import.original_name, "Bar");
    }

    #[test]
    fn test_glob_does_not_overwrite_existing() {
        // Glob should not overwrite existing items with the same name
        let mut tree = create_test_tree_with_modules();

        // Root already has an item called "Bar"
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_item(ModuleItem {
                name: "Bar".to_string(),
                kind: ItemKind::Function,
                visibility: Visibility::Public,
                node_id: NodeId(999), // Different from foo::Bar (NodeId(100))
            });
            root.add_import(ImportEntry {
                local_name: "*".to_string(),
                source_path: vec!["foo".to_string()],
                original_name: "*".to_string(),
                is_glob: true,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        tree.resolve_imports().unwrap();

        // Root should still have its own Bar, not foo::Bar
        let root = tree.get(&ModuleId::root()).unwrap();
        let own_bar = root.items.iter().find(|i| i.name == "Bar").unwrap();
        assert_eq!(own_bar.node_id, NodeId(999));

        // Glob should not have added another Bar import
        let bar_imports: Vec<_> = root
            .imports
            .iter()
            .filter(|i| i.local_name == "Bar" && !i.is_glob)
            .collect();
        assert!(
            bar_imports.is_empty(),
            "Glob should not overwrite existing items"
        );
    }

    #[test]
    fn test_glob_import_nested_module() {
        // Test: use outer::inner::* imports items from nested module
        let mut tree = ModuleTree::new();

        // Create outer module
        let outer_id = ModuleId::root().join("outer");
        {
            let outer = tree.get_or_create(outer_id.clone());
            outer.visibility = Visibility::Public;
        }

        // Create inner module as child of outer
        let inner_id = outer_id.join("inner");
        {
            let inner = tree.get_or_create(inner_id.clone());
            inner.visibility = Visibility::Public;
            inner.add_item(ModuleItem {
                name: "NestedStruct".to_string(),
                kind: ItemKind::Struct,
                visibility: Visibility::Public,
                node_id: NodeId(100),
            });
            inner.add_item(ModuleItem {
                name: "nested_fn".to_string(),
                kind: ItemKind::Function,
                visibility: Visibility::Public,
                node_id: NodeId(101),
            });
            inner.add_item(ModuleItem {
                name: "private_item".to_string(),
                kind: ItemKind::Function,
                visibility: Visibility::Private,
                node_id: NodeId(102),
            });
        }

        // Register inner as child of outer
        if let Some(outer) = tree.get_mut(&outer_id) {
            outer.add_child("inner".to_string(), inner_id.clone());
        }

        // Add glob import from nested module: use outer::inner::*
        {
            let root = tree.get_mut(&ModuleId::root()).unwrap();
            root.add_import(ImportEntry {
                local_name: "*".to_string(),
                source_path: vec!["outer".to_string(), "inner".to_string()],
                original_name: "*".to_string(),
                is_glob: true,
                is_reexport: false,
                is_module_import: false,
                resolved: None,
            });
        }

        // Resolve imports
        let result = tree.resolve_imports();
        assert!(result.is_ok(), "Nested glob import should succeed");

        // Verify public items were imported
        let root = tree.get(&ModuleId::root()).unwrap();

        let struct_import = root
            .imports
            .iter()
            .find(|i| i.local_name == "NestedStruct" && !i.is_glob);
        assert!(
            struct_import.is_some(),
            "NestedStruct should be imported via glob"
        );
        assert_eq!(struct_import.unwrap().resolved, Some(NodeId(100)));

        let fn_import = root
            .imports
            .iter()
            .find(|i| i.local_name == "nested_fn" && !i.is_glob);
        assert!(fn_import.is_some(), "nested_fn should be imported via glob");
        assert_eq!(fn_import.unwrap().resolved, Some(NodeId(101)));

        // Verify private item was NOT imported
        let private_import = root.imports.iter().find(|i| i.local_name == "private_item");
        assert!(
            private_import.is_none(),
            "Private items should not be imported via glob"
        );
    }
}
