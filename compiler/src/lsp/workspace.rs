//! Multi-file workspace support for LSP
//!
//! Provides workspace-wide features including:
//! - File discovery and indexing
//! - Dependency graph tracking
//! - Cross-file symbol resolution
//! - Module path resolution

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Item, ModuleId};
use crate::common::Span;
use crate::lexer;
use crate::parser;
use crate::resolve::{DefId, DefKind, Symbol, SymbolTable};

/// Represents a file in the workspace
#[derive(Debug)]
pub struct WorkspaceFile {
    /// File path
    pub path: PathBuf,
    /// URI for LSP
    pub uri: Url,
    /// Module path (e.g., ["std", "collections"])
    pub module_path: Vec<String>,
    /// Parsed AST (cached)
    pub ast: Option<Ast>,
    /// Symbol table for this file
    pub symbols: Option<SymbolTable>,
    /// Files this file imports
    pub imports: HashSet<PathBuf>,
    /// Files that import this file
    pub dependents: HashSet<PathBuf>,
    /// Last modification time
    pub modified: std::time::SystemTime,
    /// Source text (cached for quick access)
    pub source: Option<String>,
}

impl WorkspaceFile {
    /// Create a new workspace file
    pub fn new(path: PathBuf, uri: Url) -> Self {
        let module_path = path_to_module(&path);
        Self {
            path,
            uri,
            module_path,
            ast: None,
            symbols: None,
            imports: HashSet::new(),
            dependents: HashSet::new(),
            modified: std::time::SystemTime::UNIX_EPOCH,
            source: None,
        }
    }
}

/// A symbol exported by a file
#[derive(Debug, Clone)]
pub struct ExportedSymbol {
    /// Symbol name
    pub name: String,
    /// Symbol kind
    pub kind: DefKind,
    /// Definition span
    pub span: Span,
    /// File that defines this symbol
    pub file: PathBuf,
    /// URI of defining file
    pub uri: Url,
    /// Module path
    pub module_path: Vec<String>,
    /// Documentation (if any)
    pub doc: Option<String>,
}

/// Cross-file dependency graph
#[derive(Debug, Default)]
pub struct DependencyGraph {
    /// File -> files it imports
    imports: HashMap<PathBuf, HashSet<PathBuf>>,
    /// File -> files that import it
    dependents: HashMap<PathBuf, HashSet<PathBuf>>,
}

impl DependencyGraph {
    /// Create a new dependency graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a dependency from source to target
    pub fn add_dependency(&mut self, source: &Path, target: &Path) {
        self.imports
            .entry(source.to_path_buf())
            .or_default()
            .insert(target.to_path_buf());
        self.dependents
            .entry(target.to_path_buf())
            .or_default()
            .insert(source.to_path_buf());
    }

    /// Get files that the given file imports
    pub fn get_imports(&self, file: &Path) -> Option<&HashSet<PathBuf>> {
        self.imports.get(file)
    }

    /// Get files that import the given file
    pub fn get_dependents(&self, file: &Path) -> Option<&HashSet<PathBuf>> {
        self.dependents.get(file)
    }

    /// Clear dependencies for a file (before re-analyzing)
    pub fn clear_file(&mut self, file: &Path) {
        // Remove from imports
        if let Some(imported) = self.imports.remove(file) {
            // Remove this file from their dependents
            for dep in imported {
                if let Some(deps) = self.dependents.get_mut(&dep) {
                    deps.remove(file);
                }
            }
        }
    }

    /// Get all files that need reanalysis when a file changes
    pub fn get_affected_files(&self, changed: &Path) -> Vec<PathBuf> {
        let mut affected = Vec::new();
        let mut visited = HashSet::new();
        let mut queue = vec![changed.to_path_buf()];

        while let Some(file) = queue.pop() {
            if visited.contains(&file) {
                continue;
            }
            visited.insert(file.clone());

            if file != changed {
                affected.push(file.clone());
            }

            if let Some(deps) = self.dependents.get(&file) {
                queue.extend(deps.iter().cloned());
            }
        }

        affected
    }
}

/// Multi-file workspace manager
#[derive(Debug)]
pub struct Workspace {
    /// Workspace root URI
    pub root_uri: Option<Url>,
    /// Workspace root path
    pub root_path: Option<PathBuf>,
    /// Indexed files
    files: HashMap<PathBuf, WorkspaceFile>,
    /// URI to path mapping
    uri_to_path: HashMap<Url, PathBuf>,
    /// Global symbol index (name -> locations)
    symbol_index: HashMap<String, Vec<ExportedSymbol>>,
    /// Dependency graph
    dependency_graph: DependencyGraph,
    /// File extensions to index
    extensions: Vec<String>,
}

impl Workspace {
    /// Create a new workspace
    pub fn new() -> Self {
        Self {
            root_uri: None,
            root_path: None,
            files: HashMap::new(),
            uri_to_path: HashMap::new(),
            symbol_index: HashMap::new(),
            dependency_graph: DependencyGraph::new(),
            extensions: vec!["d".to_string(), "sio".to_string()],
        }
    }

    /// Initialize workspace with a root URI
    pub fn initialize(&mut self, root_uri: Option<Url>) {
        self.root_uri = root_uri.clone();
        self.root_path = root_uri.as_ref().and_then(|u| u.to_file_path().ok());
    }

    /// Scan workspace to discover all source files
    pub fn scan_workspace(&mut self) -> Vec<PathBuf> {
        let mut discovered = Vec::new();

        if let Some(ref root) = self.root_path {
            self.scan_directory(root, &mut discovered);
        }

        discovered
    }

    /// Recursively scan a directory for source files
    fn scan_directory(&mut self, dir: &Path, discovered: &mut Vec<PathBuf>) {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_dir() {
                    // Skip hidden directories and common non-source directories
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !name.starts_with('.') && name != "target" && name != "node_modules" {
                        self.scan_directory(&path, discovered);
                    }
                } else if self.is_source_file(&path) {
                    discovered.push(path.clone());
                    self.add_file(path);
                }
            }
        }
    }

    /// Check if a path is a source file
    fn is_source_file(&self, path: &Path) -> bool {
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| self.extensions.iter().any(|e| e == ext))
            .unwrap_or(false)
    }

    /// Add a file to the workspace
    pub fn add_file(&mut self, path: PathBuf) -> Option<&WorkspaceFile> {
        if let Ok(uri) = Url::from_file_path(&path) {
            self.uri_to_path.insert(uri.clone(), path.clone());
            self.files
                .entry(path.clone())
                .or_insert_with(|| WorkspaceFile::new(path.clone(), uri));
            self.files.get(&path)
        } else {
            None
        }
    }

    /// Get a file by path
    pub fn get_file(&self, path: &Path) -> Option<&WorkspaceFile> {
        self.files.get(path)
    }

    /// Get a file by URI
    pub fn get_file_by_uri(&self, uri: &Url) -> Option<&WorkspaceFile> {
        self.uri_to_path.get(uri).and_then(|p| self.files.get(p))
    }

    /// Get mutable file by path
    pub fn get_file_mut(&mut self, path: &Path) -> Option<&mut WorkspaceFile> {
        self.files.get_mut(path)
    }

    /// Get mutable file by URI
    pub fn get_file_by_uri_mut(&mut self, uri: &Url) -> Option<&mut WorkspaceFile> {
        if let Some(path) = self.uri_to_path.get(uri).cloned() {
            self.files.get_mut(&path)
        } else {
            None
        }
    }

    /// Remove a file from the workspace
    pub fn remove_file(&mut self, path: &Path) {
        if let Some(file) = self.files.remove(path) {
            self.uri_to_path.remove(&file.uri);
            self.dependency_graph.clear_file(path);

            // Remove symbols from index
            self.symbol_index.retain(|_, symbols| {
                symbols.retain(|s| s.file != path);
                !symbols.is_empty()
            });
        }
    }

    /// Update a file's content and re-index
    pub fn update_file(&mut self, uri: &Url, source: String) {
        if let Some(path) = self.uri_to_path.get(uri).cloned() {
            // Clear old dependencies
            self.dependency_graph.clear_file(&path);

            // Remove old symbols from index
            self.symbol_index.retain(|_, symbols| {
                symbols.retain(|s| s.file != path);
                !symbols.is_empty()
            });

            // Parse and analyze
            if let Some(file) = self.files.get_mut(&path) {
                file.source = Some(source.clone());
                file.modified = std::time::SystemTime::now();

                // Parse the file
                if let Ok(tokens) = lexer::lex(&source) {
                    if let Ok(ast) = parser::parse(&tokens, &source) {
                        // Extract exports and build symbol index
                        self.index_file_symbols(&path, &ast, &source);
                        file.ast = Some(ast);
                    }
                }
            }
        }
    }

    /// Index symbols from a file
    fn index_file_symbols(&mut self, path: &Path, ast: &Ast, source: &str) {
        let uri = Url::from_file_path(path).unwrap_or_else(|_| Url::parse("file:///").unwrap());
        let module_path = path_to_module(path);

        for item in &ast.items {
            let (name, kind, span, doc) = match item {
                Item::Function(f) => {
                    let doc = extract_doc_comment(source, f.span.start);
                    let kind = if f.modifiers.is_kernel {
                        DefKind::Kernel
                    } else {
                        DefKind::Function
                    };
                    (f.name.clone(), kind, f.span.clone(), doc)
                }
                Item::Struct(s) => {
                    let doc = extract_doc_comment(source, s.span.start);
                    (
                        s.name.clone(),
                        DefKind::Struct {
                            is_linear: s.modifiers.linear,
                            is_affine: s.modifiers.affine,
                        },
                        s.span.clone(),
                        doc,
                    )
                }
                Item::Enum(e) => {
                    let doc = extract_doc_comment(source, e.span.start);
                    (
                        e.name.clone(),
                        DefKind::Enum {
                            is_linear: e.modifiers.linear,
                            is_affine: e.modifiers.affine,
                        },
                        e.span.clone(),
                        doc,
                    )
                }
                Item::TypeAlias(t) => {
                    let doc = extract_doc_comment(source, t.span.start);
                    (t.name.clone(), DefKind::TypeAlias, t.span.clone(), doc)
                }
                Item::Effect(e) => {
                    let doc = extract_doc_comment(source, e.span.start);
                    (e.name.clone(), DefKind::Effect, e.span.clone(), doc)
                }
                Item::Trait(t) => {
                    let doc = extract_doc_comment(source, t.span.start);
                    (t.name.clone(), DefKind::Trait, t.span.clone(), doc)
                }
                Item::Global(g) if g.is_const => {
                    let doc = extract_doc_comment(source, g.span.start);
                    // Extract name from pattern
                    let name = match &g.pattern {
                        crate::ast::Pattern::Binding { name, .. } => name.clone(),
                        _ => continue, // Skip non-simple patterns for now
                    };
                    (name, DefKind::Const, g.span.clone(), doc)
                }
                Item::Module(m) => {
                    let doc = extract_doc_comment(source, m.span.start);
                    (m.name.clone(), DefKind::Module, m.span.clone(), doc)
                }
                _ => continue,
            };

            let symbol = ExportedSymbol {
                name: name.clone(),
                kind,
                span,
                file: path.to_path_buf(),
                uri: uri.clone(),
                module_path: module_path.clone(),
                doc,
            };

            self.symbol_index.entry(name).or_default().push(symbol);
        }
    }

    /// Resolve a module path to a file
    pub fn resolve_module_path(&self, module_path: &[String]) -> Option<PathBuf> {
        let root = self.root_path.as_ref()?;

        // Try different file patterns
        // 1. path/to/module.d
        // 2. path/to/module/mod.d
        // 3. path/to/module.sio
        // 4. path/to/module/mod.sio

        let path_str = module_path.join(std::path::MAIN_SEPARATOR_STR);

        for ext in &self.extensions {
            // Try direct file
            let direct = root.join(format!("{}.{}", path_str, ext));
            if direct.exists() {
                return Some(direct);
            }

            // Try mod file in directory
            let mod_file = root.join(&path_str).join(format!("mod.{}", ext));
            if mod_file.exists() {
                return Some(mod_file);
            }
        }

        None
    }

    /// Look up a symbol across the workspace
    pub fn lookup_symbol(&self, name: &str) -> Vec<&ExportedSymbol> {
        self.symbol_index
            .get(name)
            .map(|v| v.iter().collect())
            .unwrap_or_default()
    }

    /// Look up a symbol in a specific module
    pub fn lookup_symbol_in_module(
        &self,
        name: &str,
        module_path: &[String],
    ) -> Option<&ExportedSymbol> {
        self.symbol_index
            .get(name)
            .and_then(|symbols| symbols.iter().find(|s| s.module_path == module_path))
    }

    /// Get all exported symbols (for workspace/symbol request)
    pub fn all_symbols(&self) -> Vec<&ExportedSymbol> {
        self.symbol_index.values().flatten().collect()
    }

    /// Search symbols by name pattern
    pub fn search_symbols(&self, query: &str) -> Vec<&ExportedSymbol> {
        let query_lower = query.to_lowercase();
        self.symbol_index
            .iter()
            .filter(|(name, _)| name.to_lowercase().contains(&query_lower))
            .flat_map(|(_, symbols)| symbols.iter())
            .collect()
    }

    /// Cross-file goto definition
    pub fn goto_definition(&self, name: &str, from_uri: &Url) -> Option<GotoDefinitionResponse> {
        let symbols = self.lookup_symbol(name);

        if symbols.is_empty() {
            return None;
        }

        if symbols.len() == 1 {
            let sym = symbols[0];
            let location = symbol_to_location(sym);
            return Some(GotoDefinitionResponse::Scalar(location));
        }

        // Multiple definitions - return all
        let locations: Vec<Location> = symbols.iter().map(|s| symbol_to_location(s)).collect();
        Some(GotoDefinitionResponse::Array(locations))
    }

    /// Cross-file find references
    pub fn find_references(&self, name: &str, include_declaration: bool) -> Vec<Location> {
        let mut locations = Vec::new();

        // Add declarations
        if include_declaration {
            if let Some(symbols) = self.symbol_index.get(name) {
                for sym in symbols {
                    locations.push(symbol_to_location(sym));
                }
            }
        }

        // Search all files for references
        for file in self.files.values() {
            if let Some(ref source) = file.source {
                // Simple text search for now
                // A proper implementation would use the AST
                for (line_idx, line) in source.lines().enumerate() {
                    if let Some(col) = line.find(name) {
                        // Verify it's a word boundary
                        let before_ok = col == 0
                            || !line.as_bytes()[col - 1].is_ascii_alphanumeric()
                                && line.as_bytes()[col - 1] != b'_';
                        let after_ok = col + name.len() >= line.len()
                            || !line.as_bytes()[col + name.len()].is_ascii_alphanumeric()
                                && line.as_bytes()[col + name.len()] != b'_';

                        if before_ok && after_ok {
                            locations.push(Location {
                                uri: file.uri.clone(),
                                range: Range {
                                    start: Position {
                                        line: line_idx as u32,
                                        character: col as u32,
                                    },
                                    end: Position {
                                        line: line_idx as u32,
                                        character: (col + name.len()) as u32,
                                    },
                                },
                            });
                        }
                    }
                }
            }
        }

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
    }

    /// Get files affected by a change
    pub fn get_affected_files(&self, changed: &Path) -> Vec<PathBuf> {
        self.dependency_graph.get_affected_files(changed)
    }

    /// Get all indexed files
    pub fn all_files(&self) -> impl Iterator<Item = &WorkspaceFile> {
        self.files.values()
    }

    /// Get workspace symbol response
    pub fn workspace_symbols(&self, query: &str) -> Vec<SymbolInformation> {
        let symbols = if query.is_empty() {
            self.all_symbols()
        } else {
            self.search_symbols(query)
        };

        symbols
            .into_iter()
            .filter_map(|sym| {
                let kind = def_kind_to_symbol_kind(&sym.kind);
                #[allow(deprecated)]
                Some(SymbolInformation {
                    name: sym.name.clone(),
                    kind,
                    tags: None,
                    deprecated: None,
                    location: symbol_to_location(sym),
                    container_name: if sym.module_path.is_empty() {
                        None
                    } else {
                        Some(sym.module_path.join("::"))
                    },
                })
            })
            .collect()
    }
}

impl Default for Workspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a file path to a module path
fn path_to_module(path: &Path) -> Vec<String> {
    let mut parts = Vec::new();

    // Get path components, excluding file extension
    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
        // Don't include "mod" as it's implicit
        if stem != "mod" {
            parts.push(stem.to_string());
        }
    }

    // Add parent directories (limited depth)
    if let Some(parent) = path.parent() {
        for component in parent.components().rev().take(5) {
            if let std::path::Component::Normal(name) = component {
                if let Some(s) = name.to_str() {
                    // Stop at common root directories
                    if s == "src" || s == "lib" || s == "examples" {
                        break;
                    }
                    parts.push(s.to_string());
                }
            }
        }
    }

    parts.reverse();
    parts
}

/// Extract doc comment before a span
fn extract_doc_comment(source: &str, offset: usize) -> Option<String> {
    if offset == 0 {
        return None;
    }

    let before = &source[..offset];
    let mut doc_lines = Vec::new();
    let mut in_doc = false;

    for line in before.lines().rev().take(20) {
        let trimmed = line.trim();

        if trimmed.starts_with("///") {
            doc_lines.push(trimmed[3..].trim());
            in_doc = true;
        } else if trimmed.starts_with("//!") {
            doc_lines.push(trimmed[3..].trim());
            in_doc = true;
        } else if in_doc && trimmed.is_empty() {
            // Empty line continues doc block
        } else if in_doc {
            break;
        } else if trimmed.is_empty() {
            // Skip whitespace before doc
        } else {
            break;
        }
    }

    if doc_lines.is_empty() {
        None
    } else {
        doc_lines.reverse();
        Some(doc_lines.join("\n"))
    }
}

/// Convert ExportedSymbol to LSP Location
fn symbol_to_location(sym: &ExportedSymbol) -> Location {
    // We need to convert span offsets to line/column
    // For now, use a simple heuristic
    Location {
        uri: sym.uri.clone(),
        range: Range {
            start: Position {
                line: 0,
                character: sym.span.start as u32,
            },
            end: Position {
                line: 0,
                character: sym.span.end as u32,
            },
        },
    }
}

/// Convert DefKind to LSP SymbolKind
fn def_kind_to_symbol_kind(kind: &DefKind) -> SymbolKind {
    match kind {
        DefKind::Function => SymbolKind::FUNCTION,
        DefKind::Variable { .. } => SymbolKind::VARIABLE,
        DefKind::Parameter { .. } => SymbolKind::VARIABLE,
        DefKind::Struct { .. } => SymbolKind::STRUCT,
        DefKind::Enum { .. } => SymbolKind::ENUM,
        DefKind::Variant => SymbolKind::ENUM_MEMBER,
        DefKind::TypeAlias => SymbolKind::TYPE_PARAMETER,
        DefKind::TypeParam => SymbolKind::TYPE_PARAMETER,
        DefKind::Effect => SymbolKind::INTERFACE,
        DefKind::EffectOp => SymbolKind::METHOD,
        DefKind::Const => SymbolKind::CONSTANT,
        DefKind::Module => SymbolKind::MODULE,
        DefKind::Trait => SymbolKind::INTERFACE,
        DefKind::Field => SymbolKind::FIELD,
        DefKind::Kernel => SymbolKind::FUNCTION,
        DefKind::BuiltinType => SymbolKind::CLASS,
        DefKind::BuiltinFunction => SymbolKind::FUNCTION,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_path_to_module() {
        let path = PathBuf::from("/project/src/utils/helper.d");
        let module = path_to_module(&path);
        assert!(module.contains(&"helper".to_string()));
    }

    #[test]
    fn test_workspace_creation() {
        let workspace = Workspace::new();
        assert!(workspace.root_uri.is_none());
        assert!(workspace.files.is_empty());
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = DependencyGraph::new();
        let file_a = PathBuf::from("/a.d");
        let file_b = PathBuf::from("/b.d");
        let file_c = PathBuf::from("/c.d");

        graph.add_dependency(&file_a, &file_b);
        graph.add_dependency(&file_b, &file_c);

        assert!(graph.get_imports(&file_a).unwrap().contains(&file_b));
        assert!(graph.get_dependents(&file_b).unwrap().contains(&file_a));

        let affected = graph.get_affected_files(&file_c);
        assert!(affected.contains(&file_b));
    }

    #[test]
    fn test_extract_doc_comment() {
        let source = r#"/// This is a doc comment
/// with multiple lines
fn foo() {}"#;

        let doc = extract_doc_comment(source, source.find("fn").unwrap());
        assert!(doc.is_some());
        assert!(doc.unwrap().contains("doc comment"));
    }
}
