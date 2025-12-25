//! Documentation model
//!
//! Represents the complete documentation for a crate.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Complete documentation for a crate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrateDoc {
    /// Crate name
    pub name: String,

    /// Crate version
    pub version: String,

    /// Crate-level documentation (//! comments)
    pub doc: Option<String>,

    /// Root module
    pub root_module: ModuleDoc,

    /// All documented items by path
    #[serde(skip)]
    pub items: BTreeMap<String, DocItem>,

    /// Search index
    pub search_index: SearchIndex,

    /// Source file information
    #[serde(skip)]
    pub source_files: Vec<SourceFile>,
}

impl CrateDoc {
    /// Create a new crate documentation
    pub fn new(name: String, version: String) -> Self {
        Self {
            name: name.clone(),
            version,
            doc: None,
            root_module: ModuleDoc::new(name.clone(), name),
            items: BTreeMap::new(),
            search_index: SearchIndex::new(),
            source_files: Vec::new(),
        }
    }
}

/// Documentation for a module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDoc {
    /// Module name
    pub name: String,

    /// Full path (e.g., "std::collections")
    pub path: String,

    /// Module documentation
    pub doc: Option<String>,

    /// Visibility
    pub visibility: Visibility,

    /// Submodules
    pub modules: Vec<ModuleDoc>,

    /// Functions
    pub functions: Vec<FunctionDoc>,

    /// Types (structs, enums, type aliases)
    pub types: Vec<TypeDoc>,

    /// Traits
    pub traits: Vec<TraitDoc>,

    /// Constants
    pub constants: Vec<ConstantDoc>,

    /// Macros
    pub macros: Vec<MacroDoc>,

    /// Re-exports
    pub reexports: Vec<ReexportDoc>,
}

impl ModuleDoc {
    /// Create a new module documentation
    pub fn new(name: String, path: String) -> Self {
        Self {
            name,
            path,
            doc: None,
            visibility: Visibility::Public,
            modules: Vec::new(),
            functions: Vec::new(),
            types: Vec::new(),
            traits: Vec::new(),
            constants: Vec::new(),
            macros: Vec::new(),
            reexports: Vec::new(),
        }
    }
}

/// Documentation for a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDoc {
    /// Function name
    pub name: String,

    /// Full path
    pub path: String,

    /// Documentation content
    pub doc: Option<String>,

    /// Parsed documentation sections
    #[serde(skip)]
    pub doc_sections: Option<super::parser::DocSections>,

    /// Visibility
    pub visibility: Visibility,

    /// Type parameters
    pub type_params: Vec<TypeParamInfo>,

    /// Parameters
    pub params: Vec<ParamInfo>,

    /// Return type
    pub return_type: TypeInfo,

    /// Effects
    pub effects: Vec<String>,

    /// Where clauses
    pub where_clauses: Vec<WhereClause>,

    /// Whether this is unsafe
    pub is_unsafe: bool,

    /// Whether this is async
    pub is_async: bool,

    /// Whether this is a const fn
    pub is_const: bool,

    /// Whether this is a kernel fn
    pub is_kernel: bool,

    /// Source location
    pub source: SourceLocation,

    /// Signature as rendered string
    pub signature: String,
}

impl FunctionDoc {
    /// Create a new function documentation
    pub fn new(name: String, path: String) -> Self {
        Self {
            name,
            path,
            doc: None,
            doc_sections: None,
            visibility: Visibility::Public,
            type_params: Vec::new(),
            params: Vec::new(),
            return_type: TypeInfo::unit(),
            effects: Vec::new(),
            where_clauses: Vec::new(),
            is_unsafe: false,
            is_async: false,
            is_const: false,
            is_kernel: false,
            source: SourceLocation::default(),
            signature: String::new(),
        }
    }
}

/// Documentation for a type (struct, enum, type alias)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDoc {
    /// Type name
    pub name: String,

    /// Full path
    pub path: String,

    /// Documentation content
    pub doc: Option<String>,

    /// Parsed documentation sections
    #[serde(skip)]
    pub doc_sections: Option<super::parser::DocSections>,

    /// Visibility
    pub visibility: Visibility,

    /// Kind of type
    pub kind: TypeKind,

    /// Type parameters
    pub type_params: Vec<TypeParamInfo>,

    /// Where clauses
    pub where_clauses: Vec<WhereClause>,

    /// Fields (for structs)
    pub fields: Vec<FieldDoc>,

    /// Variants (for enums)
    pub variants: Vec<VariantDoc>,

    /// Methods (inherent impls)
    pub methods: Vec<FunctionDoc>,

    /// Trait implementations
    pub trait_impls: Vec<TraitImplDoc>,

    /// Type modifiers
    pub modifiers: TypeModifiers,

    /// Source location
    pub source: SourceLocation,
}

impl TypeDoc {
    /// Create a new type documentation
    pub fn new(name: String, path: String, kind: TypeKind) -> Self {
        Self {
            name,
            path,
            doc: None,
            doc_sections: None,
            visibility: Visibility::Public,
            kind,
            type_params: Vec::new(),
            where_clauses: Vec::new(),
            fields: Vec::new(),
            variants: Vec::new(),
            methods: Vec::new(),
            trait_impls: Vec::new(),
            modifiers: TypeModifiers::default(),
            source: SourceLocation::default(),
        }
    }
}

/// Type modifiers (linear/affine)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TypeModifiers {
    pub linear: bool,
    pub affine: bool,
}

/// Kind of type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TypeKind {
    Struct,
    Enum,
    Union,
    TypeAlias,
}

impl TypeKind {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            TypeKind::Struct => "struct",
            TypeKind::Enum => "enum",
            TypeKind::Union => "union",
            TypeKind::TypeAlias => "type",
        }
    }
}

/// Documentation for a struct field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDoc {
    /// Field name
    pub name: String,

    /// Field type
    pub ty: TypeInfo,

    /// Documentation
    pub doc: Option<String>,

    /// Visibility
    pub visibility: Visibility,
}

/// Documentation for an enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantDoc {
    /// Variant name
    pub name: String,

    /// Documentation
    pub doc: Option<String>,

    /// Fields (for tuple/struct variants)
    pub fields: Vec<FieldDoc>,

    /// Discriminant value (if explicit)
    pub discriminant: Option<String>,
}

/// Documentation for a trait
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitDoc {
    /// Trait name
    pub name: String,

    /// Full path
    pub path: String,

    /// Documentation content
    pub doc: Option<String>,

    /// Parsed documentation sections
    #[serde(skip)]
    pub doc_sections: Option<super::parser::DocSections>,

    /// Visibility
    pub visibility: Visibility,

    /// Type parameters
    pub type_params: Vec<TypeParamInfo>,

    /// Super traits
    pub super_traits: Vec<String>,

    /// Associated types
    pub assoc_types: Vec<AssocTypeDoc>,

    /// Associated constants
    pub assoc_consts: Vec<AssocConstDoc>,

    /// Required methods
    pub required_methods: Vec<FunctionDoc>,

    /// Provided methods (with default impl)
    pub provided_methods: Vec<FunctionDoc>,

    /// Known implementors
    pub implementors: Vec<String>,

    /// Source location
    pub source: SourceLocation,
}

impl TraitDoc {
    /// Create a new trait documentation
    pub fn new(name: String, path: String) -> Self {
        Self {
            name,
            path,
            doc: None,
            doc_sections: None,
            visibility: Visibility::Public,
            type_params: Vec::new(),
            super_traits: Vec::new(),
            assoc_types: Vec::new(),
            assoc_consts: Vec::new(),
            required_methods: Vec::new(),
            provided_methods: Vec::new(),
            implementors: Vec::new(),
            source: SourceLocation::default(),
        }
    }
}

/// Associated type documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssocTypeDoc {
    pub name: String,
    pub doc: Option<String>,
    pub bounds: Vec<String>,
    pub default: Option<TypeInfo>,
}

/// Associated constant documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssocConstDoc {
    pub name: String,
    pub doc: Option<String>,
    pub ty: TypeInfo,
    pub default: Option<String>,
}

/// Trait implementation documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraitImplDoc {
    /// Implemented trait
    pub trait_path: String,

    /// Type parameters for the impl
    pub type_params: Vec<TypeParamInfo>,

    /// Where clauses
    pub where_clauses: Vec<WhereClause>,

    /// Whether this is a blanket impl
    pub is_blanket: bool,

    /// Implemented methods
    pub methods: Vec<FunctionDoc>,

    /// Associated type values
    pub assoc_types: Vec<(String, TypeInfo)>,
}

/// Documentation for a constant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstantDoc {
    pub name: String,
    pub path: String,
    pub doc: Option<String>,
    pub visibility: Visibility,
    pub ty: TypeInfo,
    pub value: Option<String>,
    pub source: SourceLocation,
}

/// Documentation for a macro
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MacroDoc {
    pub name: String,
    pub path: String,
    pub doc: Option<String>,
    pub visibility: Visibility,
    pub syntax: String,
    pub source: SourceLocation,
}

/// Re-export documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReexportDoc {
    pub name: String,
    pub original_path: String,
    pub doc: Option<String>,
    pub visibility: Visibility,
}

/// Type parameter info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeParamInfo {
    pub name: String,
    pub bounds: Vec<String>,
    pub default: Option<TypeInfo>,
}

/// Parameter info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamInfo {
    pub name: String,
    pub ty: TypeInfo,
    pub is_self: bool,
    pub self_kind: Option<SelfKind>,
    pub is_mut: bool,
}

/// Self parameter kind
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelfKind {
    Value,  // self
    Ref,    // &self
    RefMut, // &!self
    Own,    // own self
}

/// Type information for display
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeInfo {
    /// Type as string
    pub display: String,

    /// Links to other documented types
    pub links: Vec<(String, String)>, // (text, path)
}

impl TypeInfo {
    /// Create unit type info
    pub fn unit() -> Self {
        Self {
            display: "unit".to_string(),
            links: Vec::new(),
        }
    }

    /// Create type info from string
    pub fn from_str(s: &str) -> Self {
        Self {
            display: s.to_string(),
            links: Vec::new(),
        }
    }
}

/// Where clause
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhereClause {
    pub ty: String,
    pub bounds: Vec<String>,
}

/// Visibility level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Visibility {
    Public,
    Crate,
    Super,
    #[default]
    Private,
}

impl Visibility {
    /// Check if visible to external users
    pub fn is_public(&self) -> bool {
        matches!(self, Visibility::Public)
    }
}

/// Source location
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceLocation {
    pub file: PathBuf,
    pub line: u32,
    pub column: u32,
}

/// Source file info
#[derive(Debug, Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub content: String,
}

/// Any documented item
#[derive(Debug, Clone)]
pub enum DocItem {
    Module(ModuleDoc),
    Function(FunctionDoc),
    Type(TypeDoc),
    Trait(TraitDoc),
    Constant(ConstantDoc),
    Macro(MacroDoc),
}

impl DocItem {
    /// Get item path
    pub fn path(&self) -> &str {
        match self {
            DocItem::Module(m) => &m.path,
            DocItem::Function(f) => &f.path,
            DocItem::Type(t) => &t.path,
            DocItem::Trait(t) => &t.path,
            DocItem::Constant(c) => &c.path,
            DocItem::Macro(m) => &m.path,
        }
    }

    /// Get item name
    pub fn name(&self) -> &str {
        match self {
            DocItem::Module(m) => &m.name,
            DocItem::Function(f) => &f.name,
            DocItem::Type(t) => &t.name,
            DocItem::Trait(t) => &t.name,
            DocItem::Constant(c) => &c.name,
            DocItem::Macro(m) => &m.name,
        }
    }

    /// Get item documentation
    pub fn doc(&self) -> Option<&str> {
        match self {
            DocItem::Module(m) => m.doc.as_deref(),
            DocItem::Function(f) => f.doc.as_deref(),
            DocItem::Type(t) => t.doc.as_deref(),
            DocItem::Trait(t) => t.doc.as_deref(),
            DocItem::Constant(c) => c.doc.as_deref(),
            DocItem::Macro(m) => m.doc.as_deref(),
        }
    }

    /// Get item visibility
    pub fn visibility(&self) -> Visibility {
        match self {
            DocItem::Module(m) => m.visibility,
            DocItem::Function(f) => f.visibility,
            DocItem::Type(t) => t.visibility,
            DocItem::Trait(t) => t.visibility,
            DocItem::Constant(c) => c.visibility,
            DocItem::Macro(m) => m.visibility,
        }
    }
}

/// Search index for documentation
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchIndex {
    /// Items indexed by name
    pub by_name: BTreeMap<String, Vec<SearchEntry>>,

    /// Full-text search terms
    pub terms: BTreeMap<String, Vec<String>>, // term -> paths
}

/// Search index entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEntry {
    /// Item path
    pub path: String,

    /// Item name
    pub name: String,

    /// Item kind
    pub kind: SearchKind,

    /// Brief description
    pub desc: String,

    /// Parent path (for methods, fields, etc.)
    pub parent: Option<String>,
}

/// Kind for search results
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchKind {
    Module,
    Function,
    Method,
    Struct,
    Enum,
    Trait,
    TypeAlias,
    Constant,
    Macro,
    Field,
    Variant,
}

impl SearchKind {
    /// Get string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            SearchKind::Module => "mod",
            SearchKind::Function => "fn",
            SearchKind::Method => "method",
            SearchKind::Struct => "struct",
            SearchKind::Enum => "enum",
            SearchKind::Trait => "trait",
            SearchKind::TypeAlias => "type",
            SearchKind::Constant => "const",
            SearchKind::Macro => "macro",
            SearchKind::Field => "field",
            SearchKind::Variant => "variant",
        }
    }
}

impl SearchIndex {
    /// Create a new search index
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an item to the search index
    pub fn add(&mut self, entry: SearchEntry) {
        // Index by name (lowercase for case-insensitive search)
        self.by_name
            .entry(entry.name.to_lowercase())
            .or_default()
            .push(entry.clone());

        // Index words in description for full-text search
        for word in entry.desc.split_whitespace() {
            let word = word.to_lowercase();
            // Only index words with at least 3 characters
            if word.len() >= 3 {
                self.terms.entry(word).or_default().push(entry.path.clone());
            }
        }

        // Also index the name itself
        let name_lower = entry.name.to_lowercase();
        self.terms
            .entry(name_lower)
            .or_default()
            .push(entry.path.clone());
    }

    /// Search for items by query
    pub fn search(&self, query: &str) -> Vec<&SearchEntry> {
        let query_lower = query.to_lowercase();
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();

        // Exact name match (highest priority)
        if let Some(entries) = self.by_name.get(&query_lower) {
            for entry in entries {
                if seen.insert(&entry.path) {
                    results.push(entry);
                }
            }
        }

        // Prefix match
        for (name, entries) in &self.by_name {
            if name.starts_with(&query_lower) && name != &query_lower {
                for entry in entries {
                    if seen.insert(&entry.path) {
                        results.push(entry);
                    }
                }
            }
        }

        // Contains match
        for (name, entries) in &self.by_name {
            if name.contains(&query_lower) && !name.starts_with(&query_lower) {
                for entry in entries {
                    if seen.insert(&entry.path) {
                        results.push(entry);
                    }
                }
            }
        }

        // Full-text search in terms
        if let Some(paths) = self.terms.get(&query_lower) {
            for path in paths {
                if seen.insert(path) {
                    // Find the entry for this path
                    for entries in self.by_name.values() {
                        for entry in entries {
                            if &entry.path == path {
                                results.push(entry);
                                break;
                            }
                        }
                    }
                }
            }
        }

        results
    }

    /// Get number of indexed items
    pub fn len(&self) -> usize {
        self.by_name.values().map(|v| v.len()).sum()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_index_add_and_search() {
        let mut index = SearchIndex::new();

        index.add(SearchEntry {
            path: "std::vec::Vec".to_string(),
            name: "Vec".to_string(),
            kind: SearchKind::Struct,
            desc: "A growable array type".to_string(),
            parent: Some("std::vec".to_string()),
        });

        index.add(SearchEntry {
            path: "std::collections::HashMap".to_string(),
            name: "HashMap".to_string(),
            kind: SearchKind::Struct,
            desc: "A hash map implementation".to_string(),
            parent: Some("std::collections".to_string()),
        });

        // Exact match
        let results = index.search("vec");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Vec");

        // Prefix match
        let results = index.search("hash");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "HashMap");

        // Contains match
        let results = index.search("map");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_type_kind_as_str() {
        assert_eq!(TypeKind::Struct.as_str(), "struct");
        assert_eq!(TypeKind::Enum.as_str(), "enum");
        assert_eq!(TypeKind::TypeAlias.as_str(), "type");
    }

    #[test]
    fn test_visibility_is_public() {
        assert!(Visibility::Public.is_public());
        assert!(!Visibility::Private.is_public());
        assert!(!Visibility::Crate.is_public());
    }
}
