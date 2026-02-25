//! Module loader and import resolver.
//!
//! Loads a root source file, resolves its imports, and returns a single AST
//! with all imported modules merged. Qualified paths are annotated with their
//! resolved module information for proper namespace handling.

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::HashMap;
use std::path::{Path as StdPath, PathBuf};

use miette::Result;

use crate::ast::Path as AstPath;
use crate::ast::*;
use crate::lexer;
use crate::parser;

fn module_loader_debug_enabled() -> bool {
    matches!(
        std::env::var("SOUNIO_DEBUG_MODULE_LOADER").ok().as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

macro_rules! module_loader_debug {
    ($($arg:tt)*) => {
        if module_loader_debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

/// Mapping from import prefix to resolved module info
#[derive(Debug, Clone)]
pub struct ImportMapping {
    /// The prefix used in source code (e.g., ["math"] for `import math`)
    pub prefix: Vec<String>,
    /// The resolved module ID
    pub module_id: ModuleId,
    /// The file path of the resolved module
    pub file_path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct ProgramLoadOptions {
    /// Optional explicit self-hosted bootstrap manifest path.
    /// When set, directory loading for `self-hosted/` uses this manifest
    /// instead of reading `SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST`.
    pub selfhost_bootstrap_manifest: Option<PathBuf>,
    /// Optional override for strict self-hosted module gating.
    /// When `None`, falls back to `SOUNIO_SELFHOST_STRICT_MODULE_GATING`.
    pub selfhost_strict_module_gating: Option<bool>,
}

pub fn load_program_ast(entry_path: &StdPath) -> Result<Ast> {
    load_program_ast_with_options(entry_path, ProgramLoadOptions::default())
}

pub fn load_program_ast_with_options(
    entry_path: &StdPath,
    options: ProgramLoadOptions,
) -> Result<Ast> {
    // Directory compilation mode: when given a directory or mod.sio,
    // concatenate all .sio files into a single flat compilation unit.
    //
    // This is intentionally different from the import-based ModuleLoader path:
    // self-hosted suites rely on "flat" visibility across the directory tree.
    if entry_path.is_dir() {
        return load_directory_ast_with_options(entry_path, &options);
    }

    let mut loader = ModuleLoader::new()?;
    let root_id = loader.load_module(entry_path)?;
    loader.into_ast(root_id)
}

/// Recursively collect all `.sio` files from a directory tree.
///
/// Note: we intentionally skip `test_*.sio` files that live in subdirectories
/// when compiling a directory as a single flat compilation unit. This keeps
/// "library/test harness" helper files from accidentally polluting the global
/// namespace (self-hosted suites rely on this).
fn collect_sio_files(dir: &StdPath, root: &StdPath, files: &mut Vec<PathBuf>) -> Result<()> {
    let is_selfhost_root = root.file_name().and_then(|s| s.to_str()) == Some("self-hosted");
    let strict_selfhost_module_gating = is_selfhost_root && selfhost_strict_module_gating_enabled();
    let entries = std::fs::read_dir(dir)
        .map_err(|e| miette::miette!("Cannot read directory {}: {}", dir.display(), e))?;
    let mut subdirs = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| miette::miette!("Directory entry error: {}", e))?;
        let path = entry.path();
        if path.is_dir() {
            // Self-hosted suite layout includes experimental subtrees that should not be pulled
            // into the "flat directory" compilation unit. Keeping the default suite "flat"
            // but excluding experimental directories lets `souc run self-hosted/` stay green
            // while those subtrees are iterated on independently.
            if is_selfhost_root {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if is_selfhost_dir_excluded(name, strict_selfhost_module_gating) {
                        module_loader_debug!(
                            "[collect_sio_files] skipping self-hosted dir '{}' (strict={})",
                            name,
                            strict_selfhost_module_gating
                        );
                        continue;
                    }
                }
            }
            subdirs.push(path);
        } else if path.extension().and_then(|x| x.to_str()) == Some("sio") {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if is_selfhost_root && file_looks_like_markdown_fence(&path) {
                    module_loader_debug!(
                        "[collect_sio_files] skipping malformed markdown-fenced source {}",
                        path.display()
                    );
                    continue;
                }

                if is_selfhost_root
                    && path.parent().is_some_and(|p| p == root)
                    && is_selfhost_root_file_excluded(name, strict_selfhost_module_gating)
                {
                    module_loader_debug!(
                        "[collect_sio_files] skipping self-hosted root file '{}' (strict={})",
                        name,
                        strict_selfhost_module_gating
                    );
                    continue;
                }

                // Skip subdirectory-local test helpers when compiling a directory as one unit.
                // Root-level test modules are still included (self-hosted harness relies on them).
                if path.parent().is_some_and(|p| p != root) {
                    if name.starts_with("test_") {
                        continue;
                    }
                }
            }
            files.push(path);
        }
    }
    subdirs.sort();
    for subdir in subdirs {
        collect_sio_files(&subdir, root, files)?;
    }
    Ok(())
}

fn selfhost_repo_root(selfhost_dir: &StdPath) -> Option<PathBuf> {
    selfhost_dir
        .canonicalize()
        .ok()
        .and_then(|canonical| canonical.parent().map(StdPath::to_path_buf))
        .or_else(|| selfhost_dir.parent().map(StdPath::to_path_buf))
}

fn resolve_manifest_path(selfhost_dir: &StdPath, raw_path: &str) -> PathBuf {
    let manifest_path = PathBuf::from(raw_path);
    if manifest_path.is_absolute() {
        return manifest_path;
    }

    let mut candidates = Vec::new();
    if let Some(repo_root) = selfhost_repo_root(selfhost_dir) {
        candidates.push(repo_root.join(&manifest_path));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(&manifest_path));
    }

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    manifest_path
}

fn resolve_manifest_entry_path(
    selfhost_dir: &StdPath,
    manifest_path: &StdPath,
    raw_entry: &str,
) -> PathBuf {
    let entry_path = PathBuf::from(raw_entry);
    if entry_path.is_absolute() {
        return entry_path;
    }

    let mut candidates = Vec::new();
    if let Some(manifest_dir) = manifest_path.parent() {
        candidates.push(manifest_dir.join(&entry_path));
    }
    if let Some(repo_root) = selfhost_repo_root(selfhost_dir) {
        candidates.push(repo_root.join(&entry_path));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(&entry_path));
    }

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    entry_path
}

fn load_selfhost_manifest_files(
    selfhost_dir: &StdPath,
    options: &ProgramLoadOptions,
) -> Result<Option<Vec<PathBuf>>> {
    let manifest_path = if let Some(explicit_manifest) = &options.selfhost_bootstrap_manifest {
        if explicit_manifest.is_absolute() {
            explicit_manifest.clone()
        } else {
            resolve_manifest_path(selfhost_dir, explicit_manifest.to_string_lossy().as_ref())
        }
    } else {
        let Some(raw_manifest_path) = std::env::var("SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST")
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
        else {
            return Ok(None);
        };
        resolve_manifest_path(selfhost_dir, &raw_manifest_path)
    };
    let manifest_content = std::fs::read_to_string(&manifest_path).map_err(|e| {
        miette::miette!(
            "Failed to read SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST at {}: {}",
            manifest_path.display(),
            e
        )
    })?;

    let mut files = Vec::new();
    let mut seen = std::collections::HashSet::<PathBuf>::new();
    for (line_no, raw_line) in manifest_content.lines().enumerate() {
        let line = match raw_line.split_once('#') {
            Some((prefix, _)) => prefix.trim(),
            None => raw_line.trim(),
        };
        if line.is_empty() {
            continue;
        }

        let resolved = resolve_manifest_entry_path(selfhost_dir, &manifest_path, line);
        if resolved.extension().and_then(|ext| ext.to_str()) != Some("sio") {
            return Err(miette::miette!(
                "Manifest {} line {} must reference a .sio file, got {}",
                manifest_path.display(),
                line_no + 1,
                resolved.display()
            ));
        }
        if !resolved.exists() {
            return Err(miette::miette!(
                "Manifest {} line {} points to missing file {}",
                manifest_path.display(),
                line_no + 1,
                resolved.display()
            ));
        }
        if resolved.is_dir() {
            return Err(miette::miette!(
                "Manifest {} line {} points to directory {}, expected .sio file",
                manifest_path.display(),
                line_no + 1,
                resolved.display()
            ));
        }

        let canonical = resolved.canonicalize().unwrap_or(resolved);
        if !seen.insert(canonical.clone()) {
            return Err(miette::miette!(
                "Manifest {} line {} has duplicate entry {}",
                manifest_path.display(),
                line_no + 1,
                canonical.display()
            ));
        }
        files.push(canonical);
    }

    if files.is_empty() {
        return Err(miette::miette!(
            "Manifest {} does not contain any .sio entries",
            manifest_path.display()
        ));
    }

    module_loader_debug!(
        "[load_directory_ast] using manifest {} with {} files",
        manifest_path.display(),
        files.len()
    );
    Ok(Some(files))
}

fn file_looks_like_markdown_fence(path: &StdPath) -> bool {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(_) => return false,
    };
    let mut idx = 0;
    while idx < bytes.len() && bytes[idx].is_ascii_whitespace() {
        idx += 1;
    }
    bytes.get(idx) == Some(&b'`')
}

fn selfhost_strict_module_gating_enabled() -> bool {
    matches!(
        std::env::var("SOUNIO_SELFHOST_STRICT_MODULE_GATING")
            .ok()
            .as_deref(),
        Some("1" | "true" | "TRUE" | "yes" | "YES" | "on" | "ON")
    )
}

fn selfhost_strict_module_gating_enabled_with_options(options: &ProgramLoadOptions) -> bool {
    options
        .selfhost_strict_module_gating
        .unwrap_or_else(selfhost_strict_module_gating_enabled)
}

fn strict_profile_list_contains(env_name: &str, needle: &str) -> bool {
    std::env::var(env_name).ok().is_some_and(|raw| {
        raw.split(',')
            .map(str::trim)
            .filter(|segment| !segment.is_empty())
            .any(|segment| segment == needle)
    })
}

fn is_selfhost_dir_excluded(name: &str, strict: bool) -> bool {
    // NOTE: `native/` was previously excluded but is now integrated for Phase 6.
    // These directories contain standalone experiments and often define symbols
    // that intentionally overlap with the core suite.
    if matches!(name, "hypercomplex" | "linker" | "bench") {
        return true;
    }
    strict && strict_profile_list_contains("SOUNIO_SELFHOST_STRICT_EXCLUDE_DIRS", name)
}

fn is_selfhost_root_file_excluded(name: &str, strict: bool) -> bool {
    strict
        && (name.starts_with("test_")
            || strict_profile_list_contains("SOUNIO_SELFHOST_STRICT_EXCLUDE_FILES", name))
}

fn is_selfhost_duplicate_symbol_allowlisted(name: &str) -> bool {
    matches!(
        name,
        "is_hyper_type"
            | "ALG_REAL"
            | "ALG_COMPLEX"
            | "ALG_QUATERNION"
            | "ALG_OCTONION"
            | "ALG_SEDENION"
            | "ALG_CLIFFORD"
            | "is_division_algebra"
            | "is_associative"
            | "has_zero_divisors"
            | "hlir_type_is_integer"
            | "hlir_type_is_signed"
            | "hlir_type_is_float"
            | "hlir_type_is_vector"
            | "hlir_type_is_hypercomplex"
    )
}

fn filter_selfhost_duplicate_items(items: &mut Vec<Item>) {
    let mut filtered = Vec::with_capacity(items.len());
    let mut seen = std::collections::HashSet::<String>::new();

    for item in std::mem::take(items) {
        let mut keep = true;
        if let Some(name) = item_name(&item) {
            if seen.contains(&name) && is_selfhost_duplicate_symbol_allowlisted(&name) {
                module_loader_debug!(
                    "[load_directory_ast] skipping allowlisted duplicate symbol '{}'",
                    name
                );
                keep = false;
            } else {
                seen.insert(name);
            }
        }

        if keep {
            filtered.push(item);
        }
    }

    *items = filtered;
}

/// Load all `.sio` files in a directory tree as a single flat compilation unit.
/// Files are sorted: subdirectories before parent (deeper first), alphabetical
/// within same depth, `mod.sio` always last within its directory.
/// This enables multi-file projects where definitions in earlier files are
/// visible to later files without explicit import statements.
fn load_directory_ast(dir: &StdPath) -> Result<Ast> {
    load_directory_ast_with_options(dir, &ProgramLoadOptions::default())
}

fn load_directory_ast_with_options(dir: &StdPath, options: &ProgramLoadOptions) -> Result<Ast> {
    let is_selfhost_root = dir.file_name().and_then(|s| s.to_str()) == Some("self-hosted");

    let mut files: Vec<PathBuf> = Vec::new();
    let mut loaded_from_manifest = false;
    if is_selfhost_root {
        if let Some(manifest_files) = load_selfhost_manifest_files(dir, options)? {
            files = manifest_files;
            loaded_from_manifest = true;
        }
    }
    if files.is_empty() {
        collect_sio_files(dir, dir, &mut files)?;
    }

    if files.is_empty() {
        return Err(miette::miette!(
            "No .sio files found in directory {}",
            dir.display()
        ));
    }

    let strict_selfhost_module_gating =
        is_selfhost_root && selfhost_strict_module_gating_enabled_with_options(options);

    if !loaded_from_manifest {
        // Sort: subdirectories before parent (deeper first), alphabetical within
        // same depth, mod.sio always last within its directory.
        files.sort_by(|a, b| {
            let a_dir = a.parent().unwrap_or(StdPath::new(""));
            let b_dir = b.parent().unwrap_or(StdPath::new(""));
            if a_dir != b_dir {
                let a_depth = a_dir.components().count();
                let b_depth = b_dir.components().count();
                if a_depth != b_depth {
                    return b_depth.cmp(&a_depth);
                }
                return a_dir.cmp(b_dir);
            }
            let a_is_mod = a.file_name().and_then(|f| f.to_str()) == Some("mod.sio");
            let b_is_mod = b.file_name().and_then(|f| f.to_str()) == Some("mod.sio");
            match (a_is_mod, b_is_mod) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.cmp(b),
            }
        });
    }

    let mut all_items = Vec::new();
    let mut all_node_spans = std::collections::HashMap::new();
    let mut next_node_id = 0u32;

    for file in &files {
        module_loader_debug!("[load_directory_ast] Parsing {}", file.display());
        let source = std::fs::read_to_string(file)
            .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;
        let tokens = lexer::lex(&source)
            .map_err(|e| miette::miette!("Lexer error in {}: {}", file.display(), e))?;
        let (mut ast, next_id) = parser::parse_with_id_start(&tokens, &source, next_node_id)
            .map_err(|e| miette::miette!("Parse error in {}: {}", file.display(), e))?;
        next_node_id = next_id;
        all_items.append(&mut ast.items);
        all_node_spans.extend(ast.node_spans);
    }

    if is_selfhost_root {
        filter_selfhost_duplicate_items(&mut all_items);
    }

    if strict_selfhost_module_gating {
        module_loader_debug!("[load_directory_ast] strict self-host gating enabled (no stub injection)");
    }

    // Resolve external imports: find `use` items that reference modules outside
    // this directory (e.g., `use lexer::*` from a parser/ directory), load them,
    // and prepend their items so types/functions are available.
    let stdlib_dir = find_stdlib_path();
    let canonical_dir = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
    let mod_sio = canonical_dir.join("mod.sio");
    let reference_path = if mod_sio.exists() {
        &mod_sio
    } else {
        &files[0]
    };

    let external_import_paths = collect_import_paths_from_items(&all_items);
    let mut external_items = Vec::new();
    let mut loaded_external = std::collections::HashSet::new();

    for import_path in &external_import_paths {
        // Skip if this import resolves within the directory itself
        let local_check = resolve_in_directory_silent(&canonical_dir, import_path);
        if local_check.is_ok() {
            continue;
        }

        let import_key = import_path.join("::");
        if loaded_external.contains(&import_key) {
            continue;
        }

        // Try to resolve externally (parent dir, then stdlib)
        if let Ok(external_file) =
            resolve_direct_or_local_import(reference_path, import_path, &stdlib_dir)
        {
            module_loader_debug!(
                "[load_directory_ast] Resolved external import '{}' -> {}",
                import_key,
                external_file.display()
            );
            loaded_external.insert(import_key);

            // Load the external module (which may itself be a directory module)
            let ext_canonical = external_file
                .canonicalize()
                .unwrap_or(external_file.clone());
            let is_mod = ext_canonical.file_name().and_then(|f| f.to_str()) == Some("mod.sio");
            let ext_ast = if is_mod {
                if let Some(ext_dir) = ext_canonical.parent() {
                    let has_siblings = std::fs::read_dir(ext_dir)
                        .map(|rd| {
                            rd.filter_map(|e| e.ok()).any(|e| {
                                e.path().extension().and_then(|x| x.to_str()) == Some("sio")
                                    && e.path() != ext_canonical
                            })
                        })
                        .unwrap_or(false);
                    if has_siblings {
                        load_directory_ast(ext_dir)?
                    } else {
                        let src = std::fs::read_to_string(&ext_canonical).map_err(|e| {
                            miette::miette!("Failed to read {}: {}", ext_canonical.display(), e)
                        })?;
                        let toks = lexer::lex(&src)?;
                        let (parsed, nid) = parser::parse_with_id_start(&toks, &src, next_node_id)?;
                        next_node_id = nid;
                        parsed
                    }
                } else {
                    continue;
                }
            } else {
                let src = std::fs::read_to_string(&ext_canonical).map_err(|e| {
                    miette::miette!("Failed to read {}: {}", ext_canonical.display(), e)
                })?;
                let toks = lexer::lex(&src)?;
                let (parsed, nid) = parser::parse_with_id_start(&toks, &src, next_node_id)?;
                next_node_id = nid;
                parsed
            };

            // Wrap the external items in a Module so `use lexer::*` etc. can find them
            let module_name = import_path.first().cloned().unwrap_or_default();
            let module_item = Item::Module(crate::ast::ModuleDef {
                id: crate::common::NodeId::dummy(),
                visibility: crate::ast::Visibility::Public,
                name: module_name,
                items: Some(ext_ast.items),
                doc: None,
                span: crate::common::Span::dummy(),
            });
            external_items.push(module_item);
            all_node_spans.extend(ext_ast.node_spans);
        }
    }

    // Prepend external module items so they're visible to the directory's own items
    external_items.append(&mut all_items);

    Ok(Ast {
        module_name: None,
        items: external_items,
        inner_doc: None,
        node_spans: all_node_spans,
    })
}

/// Extract import paths from a list of items (for use in load_directory_ast).
fn collect_import_paths_from_items(items: &[Item]) -> Vec<Vec<String>> {
    items
        .iter()
        .filter_map(|item| match item {
            Item::Import(import_def) => {
                let full_path = &import_def.path.segments;
                let module_path = match &import_def.items {
                    Some(_) => full_path.clone(),
                    None if full_path.len() > 1 => full_path[..full_path.len() - 1].to_vec(),
                    None => full_path.clone(),
                };
                Some(module_path)
            }
            _ => None,
        })
        .collect()
}

struct ModuleLoader {
    next_node_id: u32,
    stdlib_dir: PathBuf,
    modules: Vec<ModuleData>,
    path_to_id: HashMap<PathBuf, usize>,
    load_stack: Vec<PathBuf>,
}

struct ModuleData {
    id: usize,
    path: PathBuf,
    ast: Ast,
    import_paths: Vec<Vec<String>>,
    /// The module ID derived from the file path
    module_id: ModuleId,
    /// Resolved import mappings for this module
    import_mappings: Vec<ImportMapping>,
    /// The import path used to load this module (for creating proper nested structure)
    import_from_path: Option<Vec<String>>,
}

impl ModuleLoader {
    fn new() -> Result<Self> {
        Ok(Self {
            next_node_id: 1,
            stdlib_dir: find_stdlib_path(),
            modules: Vec::new(),
            path_to_id: HashMap::new(),
            load_stack: Vec::new(),
        })
    }

    fn load_module(&mut self, path: &StdPath) -> Result<usize> {
        let canonical = path.canonicalize().map_err(|e| {
            miette::miette!("Failed to resolve module path {}: {}", path.display(), e)
        })?;

        if let Some(existing) = self.path_to_id.get(&canonical) {
            return Ok(*existing);
        }

        if self.load_stack.contains(&canonical) {
            return Err(miette::miette!(
                "Circular import detected: {}",
                canonical.display()
            ));
        }

        self.load_stack.push(canonical.clone());

        // Directory import mode: when importing a mod.sio that has sibling .sio files,
        // load all files in the directory as a single flat module (same as directory mode).
        let is_mod_sio = canonical.file_name().and_then(|f| f.to_str()) == Some("mod.sio");
        let sibling_count = if is_mod_sio {
            canonical
                .parent()
                .and_then(|dir| std::fs::read_dir(dir).ok())
                .map(|rd| {
                    rd.filter_map(|e| e.ok())
                        .filter(|e| {
                            e.path().extension().and_then(|x| x.to_str()) == Some("sio")
                                && e.path() != canonical
                        })
                        .count()
                })
                .unwrap_or(0)
        } else {
            0
        };

        let mut ast = if is_mod_sio && sibling_count > 0 {
            let dir = canonical.parent().unwrap();
            load_directory_ast(dir)?
        } else {
            let source = std::fs::read_to_string(&canonical)
                .map_err(|e| miette::miette!("Failed to read {}: {}", canonical.display(), e))?;
            let tokens = lexer::lex(&source)?;
            let (mut parsed_ast, next_id) =
                parser::parse_with_id_start(&tokens, &source, self.next_node_id)?;
            self.next_node_id = next_id;

            // Resolve file-based module declarations (`mod foo;`)
            self.fill_file_modules(&mut parsed_ast.items, &canonical)?;
            parsed_ast
        };

        // Create module ID from file path
        let module_id = ModuleId::from_file_path(&canonical);

        let import_count = ast
            .items
            .iter()
            .filter(|i| matches!(i, Item::Import(_)))
            .count();
        module_loader_debug!(
            "[module_loader] Loaded {}: {} items, {} Import items",
            canonical.display(),
            ast.items.len(),
            import_count
        );
        for (idx, item) in ast.items.iter().enumerate().take(5) {
            let kind = match item {
                Item::Import(i) => format!("Import({:?})", i.path.segments),
                Item::Function(f) => format!("Function({})", f.name),
                Item::Struct(s) => format!("Struct({})", s.name),
                Item::Enum(e) => format!("Enum({})", e.name),
                Item::Module(m) => format!("Module({})", m.name),
                Item::Global(_) => "Global".to_string(),
                _ => "Other".to_string(),
            };
            module_loader_debug!("[module_loader]   item[{}] = {}", idx, kind);
        }
        let import_paths = collect_import_paths(&ast);
        module_loader_debug!(
            "[module_loader]   Resolved import_paths: {:?}",
            import_paths
        );

        // Build import mappings (will be populated after resolving imports)
        let mut import_mappings = Vec::new();
        for import_path in &import_paths {
            let resolve_result = resolve_import_path(&canonical, import_path, &self.stdlib_dir);
            module_loader_debug!(
                "[module_loader]   Resolving {:?} -> {:?}",
                import_path,
                resolve_result
            );
            if let Ok(import_file) = resolve_result {
                let imported_module_id = ModuleId::from_file_path(&import_file);
                import_mappings.push(ImportMapping {
                    prefix: import_path.clone(),
                    module_id: imported_module_id,
                    file_path: import_file,
                });
            }
        }

        // Annotate paths with module information instead of stripping them
        let module_prefixes = module_prefixes(&import_paths, &canonical);
        annotate_paths_in_ast(&mut ast, &module_prefixes, &module_id, &import_mappings);

        let id = self.modules.len();
        self.modules.push(ModuleData {
            id,
            path: canonical.clone(),
            ast,
            import_paths: import_paths.clone(),
            module_id,
            import_mappings,
            import_from_path: None, // Will be set when loaded via import
        });
        self.path_to_id.insert(canonical.clone(), id);

        let import_paths_owned = import_paths;
        module_loader_debug!(
            "[load_module] Processing {} imports for {}",
            import_paths_owned.len(),
            canonical.display()
        );
        for import_path in &import_paths_owned {
            let import_file = resolve_import_path(&canonical, import_path, &self.stdlib_dir)?;
            module_loader_debug!(
                "[load_module]   Loading import {:?} -> {}",
                import_path,
                import_file.display()
            );
            let loaded_id = self.load_module(&import_file)?;
            module_loader_debug!(
                "[load_module]   Loaded as module {}, total modules now: {}",
                loaded_id,
                self.modules.len()
            );
            // Track the import path used to load this module
            if let Some(module_data) = self.modules.get_mut(loaded_id) {
                if module_data.import_from_path.is_none() {
                    module_data.import_from_path = Some(import_path.clone());
                }
            }
        }

        self.load_stack.pop();

        Ok(id)
    }

    fn into_ast(mut self, root_id: usize) -> Result<Ast> {
        let root_module_name = self
            .modules
            .get(root_id)
            .and_then(|m| m.ast.module_name.clone());

        let mut items = Vec::new();
        let mut node_spans = std::collections::HashMap::new();

        let mut defined: HashMap<String, PathBuf> = HashMap::new();

        // Process imported modules FIRST so they're defined before root's import
        // statements reference them (the resolver processes items sequentially).
        for (idx, module) in self.modules.iter_mut().enumerate() {
            if idx == root_id {
                continue; // Skip root, handle after imports
            } else {
                // Imported module: wrap in nested Module items based on import path
                let import_path = module.import_from_path.clone().unwrap_or_else(|| {
                    // Fallback: derive from file name
                    let name = module
                        .path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unnamed")
                        .to_string();
                    vec![name]
                });

                if import_path.is_empty() {
                    continue;
                }

                // Create nested module structure for multi-level imports
                // e.g., import_path ["utils", "strings"] creates:
                // module utils { module strings { ... } }
                let inner_items = std::mem::take(&mut module.ast.items);
                let wrapped =
                    create_nested_modules(&mut self.next_node_id, &import_path, inner_items);

                // Merge top-level module into items, or create if new
                let top_name = &import_path[0];
                if let Some(prev_path) = defined.get(top_name) {
                    // Module already exists - need to merge
                    // Find existing module and merge
                    let existing = items
                        .iter_mut()
                        .find(|item| matches!(item, Item::Module(m) if &m.name == top_name));
                    if let Some(Item::Module(existing_mod)) = existing {
                        if let (Some(existing_items), Item::Module(new_mod)) =
                            (&mut existing_mod.items, &wrapped)
                        {
                            if let Some(new_items) = &new_mod.items {
                                existing_items.extend(new_items.clone());
                            }
                        }
                    }
                } else {
                    defined.insert(top_name.clone(), module.path.clone());
                    items.push(wrapped);
                }
            }

            node_spans.extend(module.ast.node_spans.drain());
        }

        // Now add root module items (after imported modules are already in `items`)
        module_loader_debug!(
            "[into_ast] {} imported module items so far, {} total modules, root_id={}",
            items.len(),
            self.modules.len(),
            root_id
        );
        if let Some(root_module) = self.modules.get_mut(root_id) {
            module_loader_debug!(
                "[into_ast] Root module has {} items",
                root_module.ast.items.len()
            );
            for item in &root_module.ast.items {
                if let Some(name) = item_name(item) {
                    if let Some(prev_path) = defined.get(&name) {
                        return Err(miette::miette!(
                            "Duplicate definition `{}` in {}\nand {}",
                            name,
                            prev_path.display(),
                            root_module.path.display()
                        ));
                    }
                    defined.insert(name, root_module.path.clone());
                }
            }
            items.append(&mut root_module.ast.items);
            node_spans.extend(root_module.ast.node_spans.drain());
        }
        module_loader_debug!("[into_ast] Final: {} items", items.len());

        Ok(Ast {
            module_name: root_module_name,
            items,
            inner_doc: None,
            node_spans,
        })
    }

    /// Resolve a file-based module declaration (`mod foo;`) to a source file path.
    /// Searches:
    ///   1. {parent_dir}/{name}.sio
    ///   2. {parent_dir}/{name}/mod.sio
    ///   3. {parent_dir}/{name}/lib.sio
    ///   4. {stdlib_dir}/{name}.sio
    ///   5. {stdlib_dir}/{name}/mod.sio
    ///   6. {stdlib_dir}/{name}/lib.sio
    fn resolve_module_file(&self, parent_dir: &StdPath, name: &str) -> Option<PathBuf> {
        let candidates = [
            parent_dir.join(format!("{}.sio", name)),
            parent_dir.join(name).join("mod.sio"),
            parent_dir.join(name).join("lib.sio"),
            self.stdlib_dir.join(format!("{}.sio", name)),
            self.stdlib_dir.join(name).join("mod.sio"),
            self.stdlib_dir.join(name).join("lib.sio"),
        ];
        candidates.into_iter().find(|p| p.exists())
    }

    /// Walk AST items and replace file-based module declarations (`mod foo;`,
    /// represented as `ModuleDef { items: None }`) with their loaded content.
    fn fill_file_modules(&mut self, items: &mut Vec<Item>, source_path: &StdPath) -> Result<()> {
        let parent_dir = source_path.parent().unwrap_or_else(|| StdPath::new("."));

        for item in items.iter_mut() {
            if let Item::Module(module_def) = item {
                if module_def.items.is_none() {
                    // File-based module: resolve and load
                    if let Some(file_path) = self.resolve_module_file(parent_dir, &module_def.name)
                    {
                        let source = std::fs::read_to_string(&file_path).map_err(|e| {
                            miette::miette!(
                                "Failed to read module file {}: {}",
                                file_path.display(),
                                e
                            )
                        })?;
                        let tokens = lexer::lex(&source)?;
                        let (mut sub_ast, next_id) =
                            parser::parse_with_id_start(&tokens, &source, self.next_node_id)?;
                        self.next_node_id = next_id;

                        // Recursively resolve nested file modules
                        self.fill_file_modules(&mut sub_ast.items, &file_path)?;

                        module_def.items = Some(sub_ast.items);
                    }
                    // If file not found, leave items as None (will be caught later)
                } else if let Some(inner_items) = &mut module_def.items {
                    // Inline module: recursively resolve any nested file modules
                    self.fill_file_modules(inner_items, source_path)?;
                }
            }
        }

        Ok(())
    }
}

/// Create nested module structure from an import path
/// e.g., ["utils", "strings"] with items creates:
/// module utils { module strings { ...items... } }
fn create_nested_modules(next_node_id: &mut u32, path: &[String], inner_items: Vec<Item>) -> Item {
    if path.is_empty() {
        // This shouldn't happen, but handle gracefully
        let id = *next_node_id;
        *next_node_id += 1;
        return Item::Module(ModuleDef {
            id: crate::common::NodeId(id),
            visibility: Visibility::Public,
            name: "unnamed".to_string(),
            items: Some(inner_items),
            doc: None,
            span: crate::common::Span::default(),
        });
    }

    // Start from innermost and work outward
    let mut current_items = inner_items;
    for name in path.iter().rev() {
        let id = *next_node_id;
        *next_node_id += 1;
        let module_def = ModuleDef {
            id: crate::common::NodeId(id),
            visibility: Visibility::Public,
            name: name.clone(),
            items: Some(current_items),
            doc: None,
            span: crate::common::Span::default(),
        };
        current_items = vec![Item::Module(module_def)];
    }

    // Return the outermost module
    current_items.pop().unwrap_or_else(|| {
        let id = *next_node_id;
        *next_node_id += 1;
        Item::Module(ModuleDef {
            id: crate::common::NodeId(id),
            visibility: Visibility::Public,
            name: "unnamed".to_string(),
            items: None,
            doc: None,
            span: crate::common::Span::default(),
        })
    })
}

fn find_stdlib_path() -> PathBuf {
    // Priority order for stdlib discovery:
    // 1. SOUNIO_STDLIB_PATH env var (most specific)
    // 2. SOUNIO_STDLIB env var (legacy)
    // 3. Relative to compiler binary: <exe_dir>/../stdlib/
    // 4. User home: ~/.sounio/stdlib/
    // 5. System paths: /usr/local/lib/sounio/stdlib, /usr/share/sounio/stdlib

    // 1. Check SOUNIO_STDLIB_PATH (new standard)
    if let Ok(path) = std::env::var("SOUNIO_STDLIB_PATH") {
        let pathbuf = PathBuf::from(path);
        if pathbuf.exists() {
            return pathbuf;
        }
    }

    // 2. Check SOUNIO_STDLIB (legacy, but still supported)
    if let Ok(path) = std::env::var("SOUNIO_STDLIB") {
        let pathbuf = PathBuf::from(path);
        if pathbuf.exists() {
            return pathbuf;
        }
    }

    // 3. Check relative to compiler binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let stdlib_candidates = [
                parent.join("stdlib"),
                parent.join("../stdlib"),
                // Development builds: <repo>/target/{debug,release}/souc -> <repo>/stdlib
                parent.join("../../stdlib"),
                parent.join("lib").join("sounio").join("stdlib"),
            ];

            for candidate in &stdlib_candidates {
                if candidate.exists() {
                    return candidate.to_path_buf();
                }
            }
        }
    }

    // 4. Check user home directory
    if let Some(home_dir) = dirs::home_dir() {
        let user_stdlib = home_dir.join(".sounio").join("stdlib");
        if user_stdlib.exists() {
            return user_stdlib;
        }
    }

    // 5. Check system-wide installation paths
    let system_paths = [
        "/usr/local/lib/sounio/stdlib",
        "/usr/lib/sounio/stdlib",
        "/usr/share/sounio/stdlib",
        "/opt/sounio/lib/stdlib",
    ];

    for path in &system_paths {
        let pathbuf = PathBuf::from(path);
        if pathbuf.exists() {
            return pathbuf;
        }
    }

    // 6. Final fallback - return the most common default even if it doesn't exist
    // This ensures the error messages can show where we looked
    PathBuf::from("/usr/share/sounio/stdlib")
}

/// Returns all the locations where the compiler searches for stdlib modules.
/// This is useful for diagnostics and the 'souc sysroot show' command.
pub fn get_stdlib_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Environment variables (if set)
    if let Ok(path) = std::env::var("SOUNIO_STDLIB_PATH") {
        paths.push(PathBuf::from(path));
    }
    if let Ok(path) = std::env::var("SOUNIO_STDLIB") {
        paths.push(PathBuf::from(path));
    }

    // Relative to compiler binary
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let stdlib_candidates = [
                parent.join("stdlib"),
                parent.join("../stdlib"),
                // Development builds: <repo>/target/{debug,release}/souc -> <repo>/stdlib
                parent.join("../../stdlib"),
                parent.join("lib").join("sounio").join("stdlib"),
            ];

            for candidate in &stdlib_candidates {
                paths.push(candidate.clone());
            }
        }
    }

    // User home directory
    if let Some(home_dir) = dirs::home_dir() {
        paths.push(home_dir.join(".sounio").join("stdlib"));
    }

    // System-wide installation paths
    let system_paths = [
        "/usr/local/lib/sounio/stdlib",
        "/usr/lib/sounio/stdlib",
        "/usr/share/sounio/stdlib",
        "/opt/sounio/lib/stdlib",
    ];

    for path in &system_paths {
        paths.push(PathBuf::from(path));
    }

    paths
}

/// Collects import paths from AST, returning the full path for each import.
/// For `use math::sedenion;`, returns `["math", "sedenion"]` - the resolution
/// code will probe the filesystem to determine if it's a module or item import.
fn collect_import_paths(ast: &Ast) -> Vec<Vec<String>> {
    let paths: Vec<Vec<String>> = ast
        .items
        .iter()
        .filter_map(|item| match item {
            Item::Import(import_def) => {
                // Always return the full path. Resolution will try:
                // 1. Full path as module (e.g., math/sedenion.sio)
                // 2. If not found, treat last segment as item (e.g., math.sio importing sedenion)
                Some(import_def.path.segments.clone())
            }
            _ => None,
        })
        .collect();
    paths
}

/// Returns valid module prefixes for path annotation.
/// For `use math::sedenion`, includes both:
/// - `["sedenion"]` - the last segment as standalone prefix
/// - `["math", "sedenion"]` - full path prefix for qualified access
fn module_prefixes(import_paths: &[Vec<String>], module_path: &StdPath) -> Vec<Vec<String>> {
    let mut prefixes = Vec::new();

    for import_path in import_paths {
        if import_path.is_empty() {
            continue;
        }
        // Always include the last segment as a prefix (e.g., `sedenion` for `use math::sedenion`)
        if let Some(last) = import_path.last() {
            prefixes.push(vec![last.clone()]);
        }
        // For multi-segment paths, also include the full path
        // This allows both `sedenion::Transform` and `math::sedenion::Transform`
        if import_path.len() > 1 {
            prefixes.push(import_path.clone());
        }
    }

    if let Some(stem) = module_path.file_stem().and_then(|s| s.to_str()) {
        prefixes.push(vec![stem.to_string()]);
    }

    prefixes
}

/// Result of import path resolution with detailed source information
#[derive(Debug, Clone)]
pub enum ImportResolution {
    /// Import found in standard library
    Stdlib(PathBuf),
    /// Import found in project/local scope
    Local(PathBuf),
    /// Import found via qualified path (e.g., std::math)
    Qualified(PathBuf, String), // (path, qualified_prefix)
}

/// Determines the scope and source of an import based on its path structure
fn determine_import_scope(import_path: &[String]) -> ImportScope {
    if import_path.is_empty() {
        return ImportScope::Invalid("empty import path".to_string());
    }

    match import_path[0].as_str() {
        "std" => ImportScope::StdlibQualified,
        _ => ImportScope::DirectOrLocal,
    }
}

/// Different scopes for import resolution
#[derive(Debug, Clone)]
enum ImportScope {
    /// Direct imports like `import linalg;` - search stdlib first
    DirectOrLocal,
    /// Qualified imports like `import std::math;` - search stdlib with prefix
    StdlibQualified,
    /// Invalid import path
    Invalid(String),
}

/// Comprehensive import path resolver with clean architecture.
/// Tries to resolve the full path as a module first (e.g., `math/sedenion.sio`).
/// If that fails and path has >1 segment, tries with last segment stripped
/// (treating it as an item import from parent module).
fn resolve_import_path(
    current_path: &StdPath,
    import_path: &[String],
    stdlib_dir: &StdPath,
) -> Result<PathBuf> {
    // Validate input
    if import_path.is_empty() {
        return Err(miette::miette!(
            "Empty import path in {}",
            current_path.display()
        ));
    }

    // Determine search strategy based on import structure
    let scope = determine_import_scope(import_path);

    // First try: resolve full path as a module
    let full_path_result = match &scope {
        ImportScope::DirectOrLocal => {
            resolve_direct_or_local_import(current_path, import_path, stdlib_dir)
        }
        ImportScope::StdlibQualified => {
            resolve_stdlib_qualified_import(current_path, import_path, stdlib_dir)
        }
        ImportScope::Invalid(msg) => {
            return Err(miette::miette!(
                "Invalid import path: {} in {}",
                msg,
                current_path.display()
            ));
        }
    };

    // If full path resolved, use it
    if full_path_result.is_ok() {
        return full_path_result;
    }

    // Fallback: if path has >1 segment, try with last segment stripped
    // This handles `use math::Transform` where Transform is an item in math module
    if import_path.len() > 1 {
        let parent_path = &import_path[..import_path.len() - 1];
        let fallback_result = match &scope {
            ImportScope::DirectOrLocal => {
                resolve_direct_or_local_import(current_path, parent_path, stdlib_dir)
            }
            ImportScope::StdlibQualified => {
                resolve_stdlib_qualified_import(current_path, parent_path, stdlib_dir)
            }
            ImportScope::Invalid(_) => unreachable!(), // Already handled above
        };

        if fallback_result.is_ok() {
            return fallback_result;
        }
    }

    // Neither worked - return the original error for better diagnostics
    full_path_result
}

/// Resolves direct imports (e.g., `import linalg;`) and local imports
fn resolve_direct_or_local_import(
    current_path: &StdPath,
    import_path: &[String],
    stdlib_dir: &StdPath,
) -> Result<PathBuf> {
    // First attempt: search in local project scope
    let local_dir = current_path
        .parent()
        .unwrap_or_else(|| StdPath::new("."))
        .to_path_buf();

    // Try local directory first
    if let Ok(result) = resolve_in_directory_silent(&local_dir, import_path) {
        return Ok(result);
    }

    // Second attempt: search in parent directory (for cross-module deps like parser importing lexer)
    if let Some(parent_dir) = local_dir.parent() {
        if let Ok(result) = resolve_in_directory_silent(parent_dir, import_path) {
            module_loader_debug!(
                "  resolved '{}' via parent dir: {}",
                import_path.join("::"),
                parent_dir.display()
            );
            return Ok(result);
        }
    }

    // Third attempt: search in stdlib as fallback
    // This allows qualified imports like `qnn::mnist::data_loader` to resolve from stdlib
    if let Ok(result) = resolve_in_directory_silent(stdlib_dir, import_path) {
        return Ok(result);
    }

    // If both fail, report error from local search for better diagnostics
    resolve_in_directory(&local_dir, import_path, "local")
}

/// Silent version of resolve_in_directory that returns None if not found
fn resolve_in_directory_silent(base_dir: &StdPath, segments: &[String]) -> Result<PathBuf, ()> {
    let path_joined = segments.join("/");

    // Check candidates without error reporting
    let candidates = vec![
        base_dir.join(format!("{}.sio", path_joined)),
        base_dir.join(&path_joined).join("mod.sio"),
        base_dir.join(&path_joined).join("lib.sio"),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    Err(())
}

/// Resolves qualified stdlib imports (e.g., `import std::math;`)
fn resolve_stdlib_qualified_import(
    current_path: &StdPath,
    import_path: &[String],
    stdlib_dir: &StdPath,
) -> Result<PathBuf> {
    if import_path.len() == 1 {
        return Err(miette::miette!(
            "Invalid import path `std` in {}",
            current_path.display()
        ));
    }

    let qualified_segments = &import_path[1..]; // Remove 'std' prefix
    resolve_in_directory(stdlib_dir, qualified_segments, "stdlib")
}

/// Core resolution logic for a specific directory
fn resolve_in_directory(
    base_dir: &StdPath,
    segments: &[String],
    source_type: &str,
) -> Result<PathBuf> {
    let mut candidates = Vec::new();
    let path_joined = segments.join("/");

    // Primary candidates: direct file matches
    candidates.push(base_dir.join(format!("{}.sio", path_joined)));
    candidates.push(base_dir.join(&path_joined).join("mod.sio"));
    candidates.push(base_dir.join(&path_joined).join("lib.sio"));

    // Case-insensitive fallback for the last segment
    if let Some(last) = segments.last() {
        let mut lowered = segments.to_vec();
        lowered[segments.len() - 1] = last.to_lowercase();
        let lowered_joined = lowered.join("/");
        candidates.push(base_dir.join(format!("{}.sio", lowered_joined)));
        candidates.push(base_dir.join(&lowered_joined).join("mod.sio"));
        candidates.push(base_dir.join(&lowered_joined).join("lib.sio"));
    }

    // Build search locations string before consuming candidates
    let search_locations: Vec<String> =
        candidates.iter().map(|p| p.display().to_string()).collect();

    // Try each candidate
    for candidate in candidates {
        if candidate.exists() {
            return Ok(candidate);
        }
    }

    // If no candidates found, provide helpful error message
    let import_name = segments.join("::");

    // Get stdlib search paths for better diagnostics
    let stdlib_paths = get_stdlib_search_paths();
    let stdlib_paths_str: Vec<String> = stdlib_paths
        .iter()
        .map(|p| {
            if p.exists() {
                format!("{} (exists)", p.display())
            } else {
                format!("{} (missing)", p.display())
            }
        })
        .collect();

    Err(miette::miette!(
        "Import `{}` not found in {}.\n\n\
         Searched in {}:\n  {}\n\n\
         Stdlib search paths:\n  {}\n\n\
         To fix this:\n\
         1. Set SOUNIO_STDLIB_PATH environment variable to your stdlib location\n\
         2. Install Sounio system-wide with: cargo install --path compiler\n\
         3. Build from source: cd compiler && cargo build --release",
        import_name,
        source_type,
        if source_type == "stdlib" {
            "stdlib directory"
        } else {
            "local directory"
        },
        search_locations.join("\n  "),
        stdlib_paths_str.join("\n  ")
    ))
}

fn item_name(item: &Item) -> Option<String> {
    match item {
        Item::Function(f) => Some(f.name.clone()),
        Item::Struct(s) => Some(s.name.clone()),
        Item::Enum(e) => Some(e.name.clone()),
        Item::Trait(t) => Some(t.name.clone()),
        Item::TypeAlias(t) => Some(t.name.clone()),
        Item::Effect(e) => Some(e.name.clone()),
        Item::Handler(h) => Some(h.name.clone()),
        Item::Global(g) => match &g.pattern {
            Pattern::Binding { name, .. } => Some(name.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn rewrite_paths_in_ast(ast: &mut Ast, prefixes: &[Vec<String>]) {
    for item in &mut ast.items {
        rewrite_paths_in_item(item, prefixes);
    }
}

/// Annotate paths in the AST with module information.
/// This sets source_module on all paths, and resolved_module on qualified paths
/// that match import prefixes.
fn annotate_paths_in_ast(
    ast: &mut Ast,
    prefixes: &[Vec<String>],
    source_module: &ModuleId,
    import_mappings: &[ImportMapping],
) {
    for item in &mut ast.items {
        annotate_paths_in_item(item, prefixes, source_module, import_mappings);
    }
}

fn annotate_paths_in_item(
    item: &mut Item,
    prefixes: &[Vec<String>],
    source_module: &ModuleId,
    import_mappings: &[ImportMapping],
) {
    match item {
        Item::Function(f) => annotate_fn_def(f, prefixes, source_module, import_mappings),
        Item::Struct(s) => annotate_struct_def(s, prefixes, source_module, import_mappings),
        Item::Enum(e) => annotate_enum_def(e, prefixes, source_module, import_mappings),
        Item::Trait(t) => annotate_trait_def(t, prefixes, source_module, import_mappings),
        Item::Impl(i) => annotate_impl_def(i, prefixes, source_module, import_mappings),
        Item::TypeAlias(t) => {
            annotate_generics(&mut t.generics, prefixes, source_module, import_mappings);
            annotate_type_expr(&mut t.ty, prefixes, source_module, import_mappings);
        }
        Item::Effect(e) => annotate_effect_def(e, prefixes, source_module, import_mappings),
        Item::Handler(h) => annotate_handler_def(h, prefixes, source_module, import_mappings),
        Item::Extern(e) => annotate_extern_block(e, prefixes, source_module, import_mappings),
        Item::Global(g) => annotate_global_def(g, prefixes, source_module, import_mappings),
        Item::OdeDef(o) => annotate_ode_def(o, prefixes, source_module, import_mappings),
        Item::PdeDef(p) => annotate_pde_def(p, prefixes, source_module, import_mappings),
        Item::CausalModel(c) => {
            annotate_causal_model_def(c, prefixes, source_module, import_mappings)
        }
        _ => {}
    }
}

fn rewrite_paths_in_item(item: &mut Item, prefixes: &[Vec<String>]) {
    match item {
        Item::Function(f) => rewrite_fn_def(f, prefixes),
        Item::Struct(s) => rewrite_struct_def(s, prefixes),
        Item::Enum(e) => rewrite_enum_def(e, prefixes),
        Item::Trait(t) => rewrite_trait_def(t, prefixes),
        Item::Impl(i) => rewrite_impl_def(i, prefixes),
        Item::TypeAlias(t) => {
            rewrite_generics(&mut t.generics, prefixes);
            rewrite_type_expr(&mut t.ty, prefixes);
        }
        Item::Effect(e) => rewrite_effect_def(e, prefixes),
        Item::Handler(h) => rewrite_handler_def(h, prefixes),
        Item::Extern(e) => rewrite_extern_block(e, prefixes),
        Item::Global(g) => rewrite_global_def(g, prefixes),
        Item::OdeDef(o) => rewrite_ode_def(o, prefixes),
        Item::PdeDef(p) => rewrite_pde_def(p, prefixes),
        Item::CausalModel(c) => rewrite_causal_model_def(c, prefixes),
        _ => {}
    }
}

fn rewrite_fn_def(def: &mut FnDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    for param in &mut def.params {
        rewrite_pattern(&mut param.pattern, prefixes);
        rewrite_type_expr(&mut param.ty, prefixes);
    }
    if let Some(ret) = &mut def.return_type {
        rewrite_type_expr(ret, prefixes);
    }
    for effect in &mut def.effects {
        rewrite_effect_ref(effect, prefixes);
    }
    rewrite_block(&mut def.body, prefixes);
}

fn rewrite_struct_def(def: &mut StructDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for field in &mut def.fields {
        rewrite_type_expr(&mut field.ty, prefixes);
    }
}

fn rewrite_enum_def(def: &mut EnumDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for variant in &mut def.variants {
        match &mut variant.data {
            VariantData::Tuple(types) => {
                for ty in types {
                    rewrite_type_expr(ty, prefixes);
                }
            }
            VariantData::Struct(fields) => {
                for field in fields {
                    rewrite_type_expr(&mut field.ty, prefixes);
                }
            }
            VariantData::Unit => {}
        }
    }
}

fn rewrite_trait_def(def: &mut TraitDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    for supertrait in &mut def.supertraits {
        rewrite_path(supertrait, prefixes);
    }
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for item in &mut def.items {
        match item {
            TraitItem::Fn(f) => rewrite_trait_fn_def(f, prefixes),
            TraitItem::Type(ty) => {
                for bound in &mut ty.bounds {
                    rewrite_path(bound, prefixes);
                }
                if let Some(default) = &mut ty.default {
                    rewrite_type_expr(default, prefixes);
                }
            }
        }
    }
}

fn rewrite_trait_fn_def(def: &mut TraitFnDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for param in &mut def.params {
        rewrite_pattern(&mut param.pattern, prefixes);
        rewrite_type_expr(&mut param.ty, prefixes);
    }
    if let Some(ret) = &mut def.return_type {
        rewrite_type_expr(ret, prefixes);
    }
    for effect in &mut def.effects {
        rewrite_effect_ref(effect, prefixes);
    }
    if let Some(body) = &mut def.default_body {
        rewrite_block(body, prefixes);
    }
}

fn rewrite_impl_def(def: &mut ImplDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    if let Some(trait_ref) = &mut def.trait_ref {
        rewrite_path(trait_ref, prefixes);
    }
    rewrite_type_expr(&mut def.target_type, prefixes);
    rewrite_where_clause(&mut def.where_clause, prefixes);
    for item in &mut def.items {
        match item {
            ImplItem::Fn(f) => rewrite_fn_def(f, prefixes),
            ImplItem::Type(t) => rewrite_type_expr(&mut t.ty, prefixes),
        }
    }
}

fn rewrite_effect_def(def: &mut EffectDef, prefixes: &[Vec<String>]) {
    for op in &mut def.operations {
        for param in &mut op.params {
            rewrite_pattern(&mut param.pattern, prefixes);
            rewrite_type_expr(&mut param.ty, prefixes);
        }
        if let Some(ret) = &mut op.return_type {
            rewrite_type_expr(ret, prefixes);
        }
    }
}

fn rewrite_handler_def(def: &mut HandlerDef, prefixes: &[Vec<String>]) {
    rewrite_generics(&mut def.generics, prefixes);
    rewrite_path(&mut def.effect, prefixes);
    for case in &mut def.cases {
        for param in &mut case.params {
            rewrite_pattern(&mut param.pattern, prefixes);
            rewrite_type_expr(&mut param.ty, prefixes);
        }
        rewrite_expr(&mut case.body, prefixes);
    }
}

fn rewrite_extern_block(block: &mut ExternBlock, prefixes: &[Vec<String>]) {
    for item in &mut block.items {
        match item {
            ExternItem::Fn(f) => {
                for param in &mut f.params {
                    rewrite_type_expr(&mut param.ty, prefixes);
                }
                if let Some(ret) = &mut f.return_type {
                    rewrite_type_expr(ret, prefixes);
                }
            }
            ExternItem::Static(s) => {
                rewrite_type_expr(&mut s.ty, prefixes);
            }
            ExternItem::Type(_) => {}
        }
    }
}

fn rewrite_global_def(def: &mut GlobalDef, prefixes: &[Vec<String>]) {
    rewrite_pattern(&mut def.pattern, prefixes);
    if let Some(ty) = &mut def.ty {
        rewrite_type_expr(ty, prefixes);
    }
    rewrite_expr(&mut def.value, prefixes);
}

fn rewrite_ode_def(def: &mut OdeDef, prefixes: &[Vec<String>]) {
    for param in &mut def.params {
        rewrite_type_expr(&mut param.ty, prefixes);
        if let Some(default) = &mut param.default {
            rewrite_expr(default, prefixes);
        }
    }
    for state in &mut def.state {
        rewrite_type_expr(&mut state.ty, prefixes);
    }
    for eq in &mut def.equations {
        rewrite_expr(&mut eq.rhs, prefixes);
    }
}

fn rewrite_pde_def(def: &mut PdeDef, prefixes: &[Vec<String>]) {
    for param in &mut def.params {
        rewrite_type_expr(&mut param.ty, prefixes);
    }
    for dim in &mut def.domain.dimensions {
        rewrite_expr(&mut dim.min, prefixes);
        rewrite_expr(&mut dim.max, prefixes);
    }
    rewrite_expr(&mut def.equation.rhs, prefixes);
    for bc in &mut def.boundary_conditions {
        rewrite_expr(&mut bc.boundary.value, prefixes);
        match &mut bc.condition {
            BoundaryConditionType::Dirichlet(expr) | BoundaryConditionType::Neumann(expr) => {
                rewrite_expr(expr, prefixes);
            }
            BoundaryConditionType::Robin { a, b, value } => {
                rewrite_expr(a, prefixes);
                rewrite_expr(b, prefixes);
                rewrite_expr(value, prefixes);
            }
            BoundaryConditionType::Periodic => {}
        }
    }
    if let Some(init) = &mut def.initial_condition {
        rewrite_expr(init, prefixes);
    }
}

fn rewrite_causal_model_def(def: &mut CausalModelDef, prefixes: &[Vec<String>]) {
    for node in &mut def.nodes {
        if let Some(ty) = &mut node.ty {
            rewrite_type_expr(ty, prefixes);
        }
    }
    for eq in &mut def.equations {
        rewrite_expr(&mut eq.rhs, prefixes);
    }
}

fn rewrite_generics(generics: &mut Generics, prefixes: &[Vec<String>]) {
    for param in &mut generics.params {
        match param {
            GenericParam::Type {
                bounds, default, ..
            } => {
                for bound in bounds {
                    rewrite_path(bound, prefixes);
                }
                if let Some(default) = default {
                    rewrite_type_expr(default, prefixes);
                }
            }
            GenericParam::Const { ty, .. } => {
                rewrite_type_expr(ty, prefixes);
            }
            GenericParam::Effect { .. } => {
                // Effect parameters don't need rewriting
            }
            GenericParam::Lifetime { .. } => {
                // Lifetime parameters don't need rewriting
            }
        }
    }
}

fn rewrite_where_clause(preds: &mut [WherePredicate], prefixes: &[Vec<String>]) {
    for pred in preds {
        rewrite_type_expr(&mut pred.ty, prefixes);
        for bound in &mut pred.bounds {
            rewrite_path(bound, prefixes);
        }
    }
}

fn rewrite_effect_ref(effect: &mut EffectRef, prefixes: &[Vec<String>]) {
    rewrite_path(&mut effect.name, prefixes);
    for arg in &mut effect.args {
        rewrite_type_expr(arg, prefixes);
    }
}

fn rewrite_pattern(pat: &mut Pattern, prefixes: &[Vec<String>]) {
    match pat {
        Pattern::Tuple(items) | Pattern::Or(items) => {
            for item in items {
                rewrite_pattern(item, prefixes);
            }
        }
        Pattern::Struct { path, fields } => {
            rewrite_path(path, prefixes);
            for (_, field_pat) in fields {
                rewrite_pattern(field_pat, prefixes);
            }
        }
        Pattern::Enum { path, patterns } => {
            rewrite_path(path, prefixes);
            if let Some(patterns) = patterns {
                for pattern in patterns {
                    rewrite_pattern(pattern, prefixes);
                }
            }
        }
        _ => {}
    }
}

fn rewrite_block(block: &mut Block, prefixes: &[Vec<String>]) {
    for stmt in &mut block.stmts {
        rewrite_stmt(stmt, prefixes);
    }
}

fn rewrite_stmt(stmt: &mut Stmt, prefixes: &[Vec<String>]) {
    match stmt {
        Stmt::Let {
            pattern, ty, value, ..
        } => {
            rewrite_pattern(pattern, prefixes);
            if let Some(ty) = ty {
                rewrite_type_expr(ty, prefixes);
            }
            if let Some(value) = value {
                rewrite_expr(value, prefixes);
            }
        }
        Stmt::Expr { expr, .. } => rewrite_expr(expr, prefixes),
        Stmt::Assign { target, value, .. } => {
            rewrite_expr(target, prefixes);
            rewrite_expr(value, prefixes);
        }
        Stmt::MacroInvocation(_) | Stmt::Empty | Stmt::LocalExtern(_) => {}
    }
}

fn rewrite_expr(expr: &mut Expr, prefixes: &[Vec<String>]) {
    match expr {
        Expr::Path { path, .. } => rewrite_path(path, prefixes),
        Expr::Binary { left, right, .. } => {
            rewrite_expr(left, prefixes);
            rewrite_expr(right, prefixes);
        }
        Expr::Unary { expr, .. } => rewrite_expr(expr, prefixes),
        Expr::Call { callee, args, .. } => {
            rewrite_expr(callee, prefixes);
            for arg in args {
                rewrite_expr(arg, prefixes);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            rewrite_expr(receiver, prefixes);
            for arg in args {
                rewrite_expr(arg, prefixes);
            }
        }
        Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
            rewrite_expr(base, prefixes);
        }
        Expr::Index { base, index, .. } => {
            rewrite_expr(base, prefixes);
            rewrite_expr(index, prefixes);
        }
        Expr::Cast { expr, ty, .. } => {
            rewrite_expr(expr, prefixes);
            rewrite_type_expr(ty, prefixes);
        }
        Expr::Block { block, .. } => rewrite_block(block, prefixes),
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            rewrite_expr(condition, prefixes);
            rewrite_block(then_branch, prefixes);
            if let Some(else_expr) = else_branch {
                rewrite_expr(else_expr, prefixes);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            rewrite_expr(scrutinee, prefixes);
            for arm in arms {
                rewrite_pattern(&mut arm.pattern, prefixes);
                if let Some(guard) = &mut arm.guard {
                    rewrite_expr(guard, prefixes);
                }
                rewrite_expr(&mut arm.body, prefixes);
            }
        }
        Expr::Loop { body, .. } => rewrite_block(body, prefixes),
        Expr::While {
            condition, body, ..
        } => {
            rewrite_expr(condition, prefixes);
            rewrite_block(body, prefixes);
        }
        Expr::For {
            pattern,
            iter,
            body,
            ..
        } => {
            rewrite_pattern(pattern, prefixes);
            rewrite_expr(iter, prefixes);
            rewrite_block(body, prefixes);
        }
        Expr::Return { value, .. } | Expr::Break { value, .. } => {
            if let Some(value) = value {
                rewrite_expr(value, prefixes);
            }
        }
        Expr::Closure {
            params,
            return_type,
            body,
            ..
        }
        | Expr::AsyncClosure {
            params,
            return_type,
            body,
            ..
        } => {
            for (_, ty) in params {
                if let Some(ty) = ty {
                    rewrite_type_expr(ty, prefixes);
                }
            }
            if let Some(ret) = return_type {
                rewrite_type_expr(ret, prefixes);
            }
            rewrite_expr(body, prefixes);
        }
        Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
            for elem in elements {
                rewrite_expr(elem, prefixes);
            }
        }
        Expr::ArrayRepeat { value, count, .. } => {
            rewrite_expr(value, prefixes);
            rewrite_expr(count, prefixes);
        }
        Expr::Range { start, end, .. } => {
            if let Some(start) = start {
                rewrite_expr(start, prefixes);
            }
            if let Some(end) = end {
                rewrite_expr(end, prefixes);
            }
        }
        Expr::StructLit { path, fields, .. } => {
            rewrite_path(path, prefixes);
            for (_, expr) in fields {
                rewrite_expr(expr, prefixes);
            }
        }
        Expr::Try { expr, .. } | Expr::Await { expr, .. } | Expr::Spawn { expr, .. } => {
            rewrite_expr(expr, prefixes);
        }
        Expr::Perform { effect, args, .. } => {
            rewrite_path(effect, prefixes);
            for arg in args {
                rewrite_expr(arg, prefixes);
            }
        }
        Expr::Handle { expr, handler, .. } => {
            rewrite_expr(expr, prefixes);
            rewrite_path(handler, prefixes);
        }
        Expr::Resume { value, .. } => rewrite_expr(value, prefixes),
        Expr::Sample { distribution, .. } => rewrite_expr(distribution, prefixes),
        Expr::AsyncBlock { block, .. } => rewrite_block(block, prefixes),
        Expr::UnsafeBlock { block, .. } => rewrite_block(block, prefixes),
        Expr::Select { arms, .. } => {
            for arm in arms {
                rewrite_expr(&mut arm.future, prefixes);
                rewrite_pattern(&mut arm.pattern, prefixes);
                if let Some(guard) = &mut arm.guard {
                    rewrite_expr(guard, prefixes);
                }
                rewrite_expr(&mut arm.body, prefixes);
            }
        }
        Expr::Join { futures, .. } => {
            for future in futures {
                rewrite_expr(future, prefixes);
            }
        }
        Expr::Do { interventions, .. } => {
            for (_, value) in interventions {
                rewrite_expr(value, prefixes);
            }
        }
        Expr::Counterfactual {
            factual,
            intervention,
            outcome,
            ..
        } => {
            rewrite_expr(factual, prefixes);
            rewrite_expr(intervention, prefixes);
            rewrite_expr(outcome, prefixes);
        }
        Expr::KnowledgeExpr {
            value,
            epsilon,
            validity,
            provenance,
            ..
        } => {
            rewrite_expr(value, prefixes);
            if let Some(eps) = epsilon {
                rewrite_expr(eps, prefixes);
            }
            if let Some(validity) = validity {
                rewrite_expr(validity, prefixes);
            }
            if let Some(provenance) = provenance {
                rewrite_expr(provenance, prefixes);
            }
        }
        Expr::Uncertain {
            value, uncertainty, ..
        } => {
            rewrite_expr(value, prefixes);
            rewrite_expr(uncertainty, prefixes);
        }
        Expr::GpuAnnotated {
            expr, annotation, ..
        } => {
            rewrite_expr(expr, prefixes);
            for (_, param_expr) in &mut annotation.params {
                rewrite_expr(param_expr, prefixes);
            }
        }
        Expr::Observe {
            data, distribution, ..
        } => {
            rewrite_expr(data, prefixes);
            rewrite_expr(distribution, prefixes);
        }
        Expr::Query {
            target,
            given,
            interventions,
            ..
        } => {
            rewrite_expr(target, prefixes);
            for g in given {
                rewrite_expr(g, prefixes);
            }
            for (_, value) in interventions {
                rewrite_expr(value, prefixes);
            }
        }
        Expr::Literal { .. }
        | Expr::Continue { .. }
        | Expr::MacroInvocation(_)
        | Expr::OntologyTerm { .. } => {}
    }
}

fn rewrite_type_expr(ty: &mut TypeExpr, prefixes: &[Vec<String>]) {
    match ty {
        TypeExpr::Named { path, args, .. } => {
            rewrite_path(path, prefixes);
            for arg in args {
                rewrite_type_expr(arg, prefixes);
            }
        }
        TypeExpr::Reference { inner, .. }
        | TypeExpr::RawPointer { inner, .. }
        | TypeExpr::Linear { inner, .. }
        | TypeExpr::Effected { inner, .. } => rewrite_type_expr(inner, prefixes),
        TypeExpr::Array { element, size } => {
            rewrite_type_expr(element, prefixes);
            if let Some(size) = size {
                rewrite_expr(size, prefixes);
            }
        }
        TypeExpr::Tuple(elements) => {
            for elem in elements {
                rewrite_type_expr(elem, prefixes);
            }
        }
        TypeExpr::Function {
            abi: _,
            params,
            return_type,
            effects,
        } => {
            for param in params {
                rewrite_type_expr(param, prefixes);
            }
            rewrite_type_expr(return_type, prefixes);
            for effect in effects {
                rewrite_effect_ref(effect, prefixes);
            }
        }
        TypeExpr::Knowledge {
            value_type,
            epsilon,
            validity,
            provenance,
        } => {
            rewrite_type_expr(value_type, prefixes);
            if let Some(epsilon) = epsilon {
                if let EpsilonValue::Expression(expr) = &mut epsilon.value {
                    rewrite_expr(expr, prefixes);
                }
                // Float and KnightianUndefined have no sub-expressions to rewrite
            }
            if let Some(validity) = validity {
                rewrite_expr(&mut validity.condition, prefixes);
            }
            if let Some(prov) = provenance
                && let Some(expr) = &mut prov.source
            {
                rewrite_expr(expr, prefixes);
            }
        }
        TypeExpr::Quantity { numeric_type, .. } => rewrite_type_expr(numeric_type, prefixes),
        TypeExpr::Tensor {
            element_type,
            shape,
        } => {
            rewrite_type_expr(element_type, prefixes);
            for dim in shape {
                if let TensorDim::Expr(expr) = dim {
                    rewrite_expr(expr, prefixes);
                }
            }
        }
        TypeExpr::Tile { element_type, .. } => rewrite_type_expr(element_type, prefixes),
        TypeExpr::Refinement {
            base_type,
            predicate,
            ..
        } => {
            rewrite_type_expr(base_type, prefixes);
            rewrite_expr(predicate, prefixes);
        }
        TypeExpr::Forall { inner, .. } => {
            rewrite_type_expr(inner, prefixes);
        }
        TypeExpr::ScientificArray { element_type, .. } => {
            rewrite_type_expr(element_type, prefixes);
        }
        TypeExpr::ScientificMatrix { element_type, .. } => {
            rewrite_type_expr(element_type, prefixes);
        }
        TypeExpr::SelfType
        | TypeExpr::Infer
        | TypeExpr::Unit
        | TypeExpr::Never
        | TypeExpr::Ontology { .. } => {}
    }
}

fn rewrite_path(path: &mut AstPath, prefixes: &[Vec<String>]) {
    if path.segments.len() <= 1 {
        return;
    }

    let module_part = &path.segments[..path.segments.len() - 1];
    if prefixes
        .iter()
        .any(|prefix| module_part == prefix.as_slice())
    {
        // Set the resolved module before stripping the prefix
        path.resolved_module = Some(ModuleId::new(module_part.to_vec()));

        if let Some(last) = path.segments.last().cloned() {
            path.segments = vec![last];
        }
    }
}

/// Annotate a path with source and resolved module information.
/// Unlike rewrite_path, this preserves the qualified path structure.
fn annotate_path(
    path: &mut AstPath,
    _prefixes: &[Vec<String>],
    source_module: &ModuleId,
    import_mappings: &[ImportMapping],
) {
    // Always set the source module
    path.source_module = Some(source_module.clone());

    // For qualified paths, try to resolve to an import mapping
    if path.segments.len() > 1 {
        let module_part = &path.segments[..path.segments.len() - 1];

        // Look for a matching import
        for mapping in import_mappings {
            if mapping.prefix == module_part
                || (mapping.prefix.len() == 1
                    && !module_part.is_empty()
                    && mapping.prefix[0] == module_part[0])
            {
                path.resolved_module = Some(mapping.module_id.clone());
                break;
            }
        }

        // If no import found, create a module ID from the path prefix
        if path.resolved_module.is_none() {
            path.resolved_module = Some(ModuleId::new(module_part.to_vec()));
        }
    }
}

// ==================== ANNOTATE HELPER FUNCTIONS ====================

fn annotate_fn_def(def: &mut FnDef, prefixes: &[Vec<String>], sm: &ModuleId, im: &[ImportMapping]) {
    annotate_generics(&mut def.generics, prefixes, sm, im);
    for param in &mut def.params {
        annotate_pattern(&mut param.pattern, prefixes, sm, im);
        annotate_type_expr(&mut param.ty, prefixes, sm, im);
    }
    if let Some(ret) = &mut def.return_type {
        annotate_type_expr(ret, prefixes, sm, im);
    }
    for effect in &mut def.effects {
        annotate_effect_ref(effect, prefixes, sm, im);
    }
    annotate_block(&mut def.body, prefixes, sm, im);
}

fn annotate_struct_def(
    def: &mut StructDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_generics(&mut def.generics, prefixes, sm, im);
    annotate_where_clause(&mut def.where_clause, prefixes, sm, im);
    for field in &mut def.fields {
        annotate_type_expr(&mut field.ty, prefixes, sm, im);
    }
}

fn annotate_enum_def(
    def: &mut EnumDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_generics(&mut def.generics, prefixes, sm, im);
    annotate_where_clause(&mut def.where_clause, prefixes, sm, im);
    for variant in &mut def.variants {
        match &mut variant.data {
            VariantData::Tuple(types) => {
                for ty in types {
                    annotate_type_expr(ty, prefixes, sm, im);
                }
            }
            VariantData::Struct(fields) => {
                for f in fields {
                    annotate_type_expr(&mut f.ty, prefixes, sm, im);
                }
            }
            VariantData::Unit => {}
        }
    }
}

fn annotate_trait_def(
    def: &mut TraitDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_generics(&mut def.generics, prefixes, sm, im);
    for st in &mut def.supertraits {
        annotate_path(st, prefixes, sm, im);
    }
    annotate_where_clause(&mut def.where_clause, prefixes, sm, im);
    for item in &mut def.items {
        match item {
            TraitItem::Fn(f) => {
                annotate_generics(&mut f.generics, prefixes, sm, im);
                annotate_where_clause(&mut f.where_clause, prefixes, sm, im);
                for p in &mut f.params {
                    annotate_pattern(&mut p.pattern, prefixes, sm, im);
                    annotate_type_expr(&mut p.ty, prefixes, sm, im);
                }
                if let Some(ret) = &mut f.return_type {
                    annotate_type_expr(ret, prefixes, sm, im);
                }
                for e in &mut f.effects {
                    annotate_effect_ref(e, prefixes, sm, im);
                }
                if let Some(body) = &mut f.default_body {
                    annotate_block(body, prefixes, sm, im);
                }
            }
            TraitItem::Type(ty) => {
                for b in &mut ty.bounds {
                    annotate_path(b, prefixes, sm, im);
                }
                if let Some(d) = &mut ty.default {
                    annotate_type_expr(d, prefixes, sm, im);
                }
            }
        }
    }
}

fn annotate_impl_def(
    def: &mut ImplDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_generics(&mut def.generics, prefixes, sm, im);
    if let Some(tr) = &mut def.trait_ref {
        annotate_path(tr, prefixes, sm, im);
    }
    annotate_type_expr(&mut def.target_type, prefixes, sm, im);
    annotate_where_clause(&mut def.where_clause, prefixes, sm, im);
    for item in &mut def.items {
        match item {
            ImplItem::Fn(f) => annotate_fn_def(f, prefixes, sm, im),
            ImplItem::Type(t) => annotate_type_expr(&mut t.ty, prefixes, sm, im),
        }
    }
}

fn annotate_effect_def(
    def: &mut EffectDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for op in &mut def.operations {
        for p in &mut op.params {
            annotate_pattern(&mut p.pattern, prefixes, sm, im);
            annotate_type_expr(&mut p.ty, prefixes, sm, im);
        }
        if let Some(ret) = &mut op.return_type {
            annotate_type_expr(ret, prefixes, sm, im);
        }
    }
}

fn annotate_handler_def(
    def: &mut HandlerDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_generics(&mut def.generics, prefixes, sm, im);
    annotate_path(&mut def.effect, prefixes, sm, im);
    for case in &mut def.cases {
        for p in &mut case.params {
            annotate_pattern(&mut p.pattern, prefixes, sm, im);
            annotate_type_expr(&mut p.ty, prefixes, sm, im);
        }
        annotate_expr(&mut case.body, prefixes, sm, im);
    }
}

fn annotate_extern_block(
    block: &mut ExternBlock,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for item in &mut block.items {
        match item {
            ExternItem::Fn(f) => {
                for p in &mut f.params {
                    annotate_type_expr(&mut p.ty, prefixes, sm, im);
                }
                if let Some(ret) = &mut f.return_type {
                    annotate_type_expr(ret, prefixes, sm, im);
                }
            }
            ExternItem::Static(s) => annotate_type_expr(&mut s.ty, prefixes, sm, im),
            ExternItem::Type(_) => {}
        }
    }
}

fn annotate_global_def(
    def: &mut GlobalDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_pattern(&mut def.pattern, prefixes, sm, im);
    if let Some(ty) = &mut def.ty {
        annotate_type_expr(ty, prefixes, sm, im);
    }
    annotate_expr(&mut def.value, prefixes, sm, im);
}

fn annotate_ode_def(
    def: &mut OdeDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for p in &mut def.params {
        annotate_type_expr(&mut p.ty, prefixes, sm, im);
        if let Some(d) = &mut p.default {
            annotate_expr(d, prefixes, sm, im);
        }
    }
    for s in &mut def.state {
        annotate_type_expr(&mut s.ty, prefixes, sm, im);
    }
    for eq in &mut def.equations {
        annotate_expr(&mut eq.rhs, prefixes, sm, im);
    }
}

fn annotate_pde_def(
    def: &mut PdeDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for p in &mut def.params {
        annotate_type_expr(&mut p.ty, prefixes, sm, im);
    }
    for dim in &mut def.domain.dimensions {
        annotate_expr(&mut dim.min, prefixes, sm, im);
        annotate_expr(&mut dim.max, prefixes, sm, im);
    }
    annotate_expr(&mut def.equation.rhs, prefixes, sm, im);
    for bc in &mut def.boundary_conditions {
        annotate_expr(&mut bc.boundary.value, prefixes, sm, im);
        match &mut bc.condition {
            BoundaryConditionType::Dirichlet(e) | BoundaryConditionType::Neumann(e) => {
                annotate_expr(e, prefixes, sm, im)
            }
            BoundaryConditionType::Robin { a, b, value } => {
                annotate_expr(a, prefixes, sm, im);
                annotate_expr(b, prefixes, sm, im);
                annotate_expr(value, prefixes, sm, im);
            }
            BoundaryConditionType::Periodic => {}
        }
    }
    if let Some(init) = &mut def.initial_condition {
        annotate_expr(init, prefixes, sm, im);
    }
}

fn annotate_causal_model_def(
    def: &mut CausalModelDef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for n in &mut def.nodes {
        if let Some(ty) = &mut n.ty {
            annotate_type_expr(ty, prefixes, sm, im);
        }
    }
    for eq in &mut def.equations {
        annotate_expr(&mut eq.rhs, prefixes, sm, im);
    }
}

fn annotate_generics(
    g: &mut Generics,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for p in &mut g.params {
        match p {
            GenericParam::Type {
                bounds, default, ..
            } => {
                for b in bounds {
                    annotate_path(b, prefixes, sm, im);
                }
                if let Some(d) = default {
                    annotate_type_expr(d, prefixes, sm, im);
                }
            }
            GenericParam::Const { ty, .. } => annotate_type_expr(ty, prefixes, sm, im),
            GenericParam::Effect { .. } => {
                // Effect parameters don't need annotation
            }
            GenericParam::Lifetime { .. } => {
                // Lifetime parameters don't need annotation
            }
        }
    }
}

fn annotate_where_clause(
    preds: &mut [WherePredicate],
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for p in preds {
        annotate_type_expr(&mut p.ty, prefixes, sm, im);
        for b in &mut p.bounds {
            annotate_path(b, prefixes, sm, im);
        }
    }
}

fn annotate_effect_ref(
    e: &mut EffectRef,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    annotate_path(&mut e.name, prefixes, sm, im);
    for a in &mut e.args {
        annotate_type_expr(a, prefixes, sm, im);
    }
}

fn annotate_pattern(
    pat: &mut Pattern,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    match pat {
        Pattern::Tuple(items) | Pattern::Or(items) => {
            for i in items {
                annotate_pattern(i, prefixes, sm, im);
            }
        }
        Pattern::Struct { path, fields } => {
            annotate_path(path, prefixes, sm, im);
            for (_, p) in fields {
                annotate_pattern(p, prefixes, sm, im);
            }
        }
        Pattern::Enum { path, patterns } => {
            annotate_path(path, prefixes, sm, im);
            if let Some(ps) = patterns {
                for p in ps {
                    annotate_pattern(p, prefixes, sm, im);
                }
            }
        }
        _ => {}
    }
}

fn annotate_block(
    block: &mut Block,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    for s in &mut block.stmts {
        annotate_stmt(s, prefixes, sm, im);
    }
}

fn annotate_stmt(stmt: &mut Stmt, prefixes: &[Vec<String>], sm: &ModuleId, im: &[ImportMapping]) {
    match stmt {
        Stmt::Let {
            pattern, ty, value, ..
        } => {
            annotate_pattern(pattern, prefixes, sm, im);
            if let Some(ty) = ty {
                annotate_type_expr(ty, prefixes, sm, im);
            }
            if let Some(v) = value {
                annotate_expr(v, prefixes, sm, im);
            }
        }
        Stmt::Expr { expr, .. } => annotate_expr(expr, prefixes, sm, im),
        Stmt::Assign { target, value, .. } => {
            annotate_expr(target, prefixes, sm, im);
            annotate_expr(value, prefixes, sm, im);
        }
        Stmt::MacroInvocation(_) | Stmt::Empty | Stmt::LocalExtern(_) => {}
    }
}

fn annotate_expr(expr: &mut Expr, prefixes: &[Vec<String>], sm: &ModuleId, im: &[ImportMapping]) {
    match expr {
        Expr::Path { path, .. } => annotate_path(path, prefixes, sm, im),
        Expr::Binary { left, right, .. } => {
            annotate_expr(left, prefixes, sm, im);
            annotate_expr(right, prefixes, sm, im);
        }
        Expr::Unary { expr, .. } => annotate_expr(expr, prefixes, sm, im),
        Expr::Call { callee, args, .. } => {
            annotate_expr(callee, prefixes, sm, im);
            for a in args {
                annotate_expr(a, prefixes, sm, im);
            }
        }
        Expr::MethodCall { receiver, args, .. } => {
            annotate_expr(receiver, prefixes, sm, im);
            for a in args {
                annotate_expr(a, prefixes, sm, im);
            }
        }
        Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
            annotate_expr(base, prefixes, sm, im)
        }
        Expr::Index { base, index, .. } => {
            annotate_expr(base, prefixes, sm, im);
            annotate_expr(index, prefixes, sm, im);
        }
        Expr::Cast { expr, ty, .. } => {
            annotate_expr(expr, prefixes, sm, im);
            annotate_type_expr(ty, prefixes, sm, im);
        }
        Expr::Block { block, .. } => annotate_block(block, prefixes, sm, im),
        Expr::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            annotate_expr(condition, prefixes, sm, im);
            annotate_block(then_branch, prefixes, sm, im);
            if let Some(e) = else_branch {
                annotate_expr(e, prefixes, sm, im);
            }
        }
        Expr::Match {
            scrutinee, arms, ..
        } => {
            annotate_expr(scrutinee, prefixes, sm, im);
            for arm in arms {
                annotate_pattern(&mut arm.pattern, prefixes, sm, im);
                if let Some(g) = &mut arm.guard {
                    annotate_expr(g, prefixes, sm, im);
                }
                annotate_expr(&mut arm.body, prefixes, sm, im);
            }
        }
        Expr::Loop { body, .. } => annotate_block(body, prefixes, sm, im),
        Expr::While {
            condition, body, ..
        } => {
            annotate_expr(condition, prefixes, sm, im);
            annotate_block(body, prefixes, sm, im);
        }
        Expr::For {
            pattern,
            iter,
            body,
            ..
        } => {
            annotate_pattern(pattern, prefixes, sm, im);
            annotate_expr(iter, prefixes, sm, im);
            annotate_block(body, prefixes, sm, im);
        }
        Expr::Return { value, .. } | Expr::Break { value, .. } => {
            if let Some(v) = value {
                annotate_expr(v, prefixes, sm, im);
            }
        }
        Expr::Closure {
            params,
            return_type,
            body,
            ..
        }
        | Expr::AsyncClosure {
            params,
            return_type,
            body,
            ..
        } => {
            for (_, ty) in params {
                if let Some(ty) = ty {
                    annotate_type_expr(ty, prefixes, sm, im);
                }
            }
            if let Some(ret) = return_type {
                annotate_type_expr(ret, prefixes, sm, im);
            }
            annotate_expr(body, prefixes, sm, im);
        }
        Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
            for e in elements {
                annotate_expr(e, prefixes, sm, im);
            }
        }
        Expr::ArrayRepeat { value, count, .. } => {
            annotate_expr(value, prefixes, sm, im);
            annotate_expr(count, prefixes, sm, im);
        }
        Expr::Range { start, end, .. } => {
            if let Some(s) = start {
                annotate_expr(s, prefixes, sm, im);
            }
            if let Some(e) = end {
                annotate_expr(e, prefixes, sm, im);
            }
        }
        Expr::StructLit { path, fields, .. } => {
            annotate_path(path, prefixes, sm, im);
            for (_, e) in fields {
                annotate_expr(e, prefixes, sm, im);
            }
        }
        Expr::Try { expr, .. } | Expr::Await { expr, .. } | Expr::Spawn { expr, .. } => {
            annotate_expr(expr, prefixes, sm, im)
        }
        Expr::Perform { effect, args, .. } => {
            annotate_path(effect, prefixes, sm, im);
            for a in args {
                annotate_expr(a, prefixes, sm, im);
            }
        }
        Expr::Handle { expr, handler, .. } => {
            annotate_expr(expr, prefixes, sm, im);
            annotate_path(handler, prefixes, sm, im);
        }
        Expr::Resume { value, .. } => annotate_expr(value, prefixes, sm, im),
        Expr::Sample { distribution, .. } => annotate_expr(distribution, prefixes, sm, im),
        Expr::AsyncBlock { block, .. } => annotate_block(block, prefixes, sm, im),
        Expr::UnsafeBlock { block, .. } => annotate_block(block, prefixes, sm, im),
        Expr::Select { arms, .. } => {
            for a in arms {
                annotate_expr(&mut a.future, prefixes, sm, im);
                annotate_pattern(&mut a.pattern, prefixes, sm, im);
                if let Some(g) = &mut a.guard {
                    annotate_expr(g, prefixes, sm, im);
                }
                annotate_expr(&mut a.body, prefixes, sm, im);
            }
        }
        Expr::Join { futures, .. } => {
            for f in futures {
                annotate_expr(f, prefixes, sm, im);
            }
        }
        Expr::Do { interventions, .. } => {
            for (_, v) in interventions {
                annotate_expr(v, prefixes, sm, im);
            }
        }
        Expr::Counterfactual {
            factual,
            intervention,
            outcome,
            ..
        } => {
            annotate_expr(factual, prefixes, sm, im);
            annotate_expr(intervention, prefixes, sm, im);
            annotate_expr(outcome, prefixes, sm, im);
        }
        Expr::KnowledgeExpr {
            value,
            epsilon,
            validity,
            provenance,
            ..
        } => {
            annotate_expr(value, prefixes, sm, im);
            if let Some(e) = epsilon {
                annotate_expr(e, prefixes, sm, im);
            }
            if let Some(v) = validity {
                annotate_expr(v, prefixes, sm, im);
            }
            if let Some(p) = provenance {
                annotate_expr(p, prefixes, sm, im);
            }
        }
        Expr::Uncertain {
            value, uncertainty, ..
        } => {
            annotate_expr(value, prefixes, sm, im);
            annotate_expr(uncertainty, prefixes, sm, im);
        }
        Expr::GpuAnnotated {
            expr, annotation, ..
        } => {
            annotate_expr(expr, prefixes, sm, im);
            for (_, p) in &mut annotation.params {
                annotate_expr(p, prefixes, sm, im);
            }
        }
        Expr::Observe {
            data, distribution, ..
        } => {
            annotate_expr(data, prefixes, sm, im);
            annotate_expr(distribution, prefixes, sm, im);
        }
        Expr::Query {
            target,
            given,
            interventions,
            ..
        } => {
            annotate_expr(target, prefixes, sm, im);
            for g in given {
                annotate_expr(g, prefixes, sm, im);
            }
            for (_, v) in interventions {
                annotate_expr(v, prefixes, sm, im);
            }
        }
        Expr::Literal { .. }
        | Expr::Continue { .. }
        | Expr::MacroInvocation(_)
        | Expr::OntologyTerm { .. } => {}
    }
}

fn annotate_type_expr(
    ty: &mut TypeExpr,
    prefixes: &[Vec<String>],
    sm: &ModuleId,
    im: &[ImportMapping],
) {
    match ty {
        TypeExpr::Named { path, args, .. } => {
            annotate_path(path, prefixes, sm, im);
            for a in args {
                annotate_type_expr(a, prefixes, sm, im);
            }
        }
        TypeExpr::Reference { inner, .. }
        | TypeExpr::RawPointer { inner, .. }
        | TypeExpr::Linear { inner, .. }
        | TypeExpr::Effected { inner, .. } => annotate_type_expr(inner, prefixes, sm, im),
        TypeExpr::Array { element, size } => {
            annotate_type_expr(element, prefixes, sm, im);
            if let Some(s) = size {
                annotate_expr(s, prefixes, sm, im);
            }
        }
        TypeExpr::Tuple(elems) => {
            for e in elems {
                annotate_type_expr(e, prefixes, sm, im);
            }
        }
        TypeExpr::Function {
            abi: _,
            params,
            return_type,
            effects,
        } => {
            for p in params {
                annotate_type_expr(p, prefixes, sm, im);
            }
            annotate_type_expr(return_type, prefixes, sm, im);
            for e in effects {
                annotate_effect_ref(e, prefixes, sm, im);
            }
        }
        TypeExpr::Knowledge {
            value_type,
            epsilon,
            validity,
            provenance,
        } => {
            annotate_type_expr(value_type, prefixes, sm, im);
            if let Some(e) = epsilon {
                if let crate::ast::EpsilonValue::Expression(expr) = &mut e.value {
                    annotate_expr(expr, prefixes, sm, im);
                }
            }
            if let Some(v) = validity {
                annotate_expr(&mut v.condition, prefixes, sm, im);
            }
            if let Some(p) = provenance
                && let Some(src) = &mut p.source
            {
                annotate_expr(src, prefixes, sm, im);
            }
        }
        TypeExpr::Quantity { numeric_type, .. } => {
            annotate_type_expr(numeric_type, prefixes, sm, im)
        }
        TypeExpr::Tensor {
            element_type,
            shape,
        } => {
            annotate_type_expr(element_type, prefixes, sm, im);
            for d in shape {
                if let TensorDim::Expr(e) = d {
                    annotate_expr(e, prefixes, sm, im);
                }
            }
        }
        TypeExpr::Tile { element_type, .. } => annotate_type_expr(element_type, prefixes, sm, im),
        TypeExpr::Refinement {
            base_type,
            predicate,
            ..
        } => {
            annotate_type_expr(base_type, prefixes, sm, im);
            annotate_expr(predicate, prefixes, sm, im);
        }
        TypeExpr::Forall { inner, .. } => {
            annotate_type_expr(inner, prefixes, sm, im);
        }
        TypeExpr::ScientificArray {
            element_type, dim, ..
        } => {
            annotate_type_expr(element_type, prefixes, sm, im);
            if let TensorDim::Expr(e) = dim {
                annotate_expr(e, prefixes, sm, im);
            }
        }
        TypeExpr::ScientificMatrix {
            element_type,
            rows,
            cols,
        } => {
            annotate_type_expr(element_type, prefixes, sm, im);
            if let TensorDim::Expr(e) = rows {
                annotate_expr(e, prefixes, sm, im);
            }
            if let TensorDim::Expr(e) = cols {
                annotate_expr(e, prefixes, sm, im);
            }
        }
        TypeExpr::SelfType
        | TypeExpr::Infer
        | TypeExpr::Unit
        | TypeExpr::Never
        | TypeExpr::Ontology { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::fs;
    use std::sync::Mutex;

    static SELFHOST_MANIFEST_ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                // SAFETY: test helper restores the prior process env value on drop.
                unsafe { std::env::set_var(self.key, prev) };
            } else {
                // SAFETY: test helper restores the prior process env state on drop.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn set_env_var(key: &'static str, value: Option<&str>) -> EnvGuard {
        let prev = std::env::var_os(key);
        if let Some(value) = value {
            // SAFETY: test helper sets env vars in a lock-guarded scope.
            unsafe { std::env::set_var(key, value) };
        } else {
            // SAFETY: test helper clears env vars in a lock-guarded scope.
            unsafe { std::env::remove_var(key) };
        }
        EnvGuard { key, prev }
    }

    #[test]
    fn selfhost_manifest_resolves_repo_relative_entries() {
        let _lock = SELFHOST_MANIFEST_ENV_LOCK.lock().expect("env lock");

        let temp = tempfile::tempdir().expect("temp dir");
        let selfhost_dir = temp.path().join("self-hosted");
        let bootstrap_dir = temp.path().join("bootstrap");
        fs::create_dir_all(&selfhost_dir).expect("create self-hosted");
        fs::create_dir_all(&bootstrap_dir).expect("create bootstrap");

        let first = selfhost_dir.join("alpha.sio");
        let second = selfhost_dir.join("beta.sio");
        fs::write(&first, "fn alpha() -> i64 { 1 }\n").expect("write alpha");
        fs::write(&second, "fn beta() -> i64 { 2 }\n").expect("write beta");

        let manifest = bootstrap_dir.join("manifest.txt");
        fs::write(
            &manifest,
            "# kernel profile\nself-hosted/alpha.sio\n\nself-hosted/beta.sio # keep beta\n",
        )
        .expect("write manifest");

        let _env_guard =
            set_env_var("SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST", Some("bootstrap/manifest.txt"));
        let files = load_selfhost_manifest_files(&selfhost_dir, &ProgramLoadOptions::default())
            .expect("load manifest")
            .expect("manifest enabled");

        assert_eq!(files.len(), 2);
        assert_eq!(
            files[0],
            first.canonicalize().expect("canonical alpha")
        );
        assert_eq!(
            files[1],
            second.canonicalize().expect("canonical beta")
        );
    }

    #[test]
    fn selfhost_manifest_preserves_file_order() {
        let _lock = SELFHOST_MANIFEST_ENV_LOCK.lock().expect("env lock");

        let temp = tempfile::tempdir().expect("temp dir");
        let selfhost_dir = temp.path().join("self-hosted");
        fs::create_dir_all(&selfhost_dir).expect("create self-hosted");

        let first = selfhost_dir.join("a.sio");
        let second = selfhost_dir.join("b.sio");
        fs::write(&first, "fn first() -> i64 { 1 }\n").expect("write a");
        fs::write(&second, "fn second() -> i64 { 2 }\n").expect("write b");

        let manifest = temp.path().join("kernel.manifest");
        fs::write(&manifest, "self-hosted/b.sio\nself-hosted/a.sio\n").expect("write manifest");

        let _env_guard = set_env_var(
            "SOUNIO_SELFHOST_BOOTSTRAP_MANIFEST",
            manifest.to_str(),
        );

        let ast = load_program_ast(&selfhost_dir).expect("load self-hosted via manifest");
        let function_names: Vec<String> = ast
            .items
            .iter()
            .filter_map(|item| match item {
                Item::Function(def) => Some(def.name.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(function_names[0], "second");
        assert_eq!(function_names[1], "first");
    }

    #[test]
    fn resolves_imports_and_rewrites_qualified_paths() {
        let dir = tempfile::tempdir().unwrap();
        let module_a = dir.path().join("a.sio");
        let module_b = dir.path().join("b.sio");

        fs::write(&module_b, "type Foo = i32;").unwrap();
        fs::write(
            &module_a,
            "import b;\n\nfn consume(foo: b.Foo) -> b.Foo { return foo }\n",
        )
        .unwrap();

        let ast = load_program_ast(&module_a).unwrap();

        let func = ast.items.iter().find_map(|item| match item {
            Item::Function(f) if f.name == "consume" => Some(f),
            _ => None,
        });
        let func = func.expect("missing consume() function");

        // Paths are now annotated with module info rather than rewritten/stripped.
        // The segments remain qualified (e.g., ["b", "Foo"]) but resolved_module is set.
        let param_ty = &func.params[0].ty;
        match param_ty {
            TypeExpr::Named { path, .. } => {
                assert_eq!(path.segments, vec!["b", "Foo"]);
                assert!(
                    path.resolved_module.is_some(),
                    "param type should have resolved_module"
                );
            }
            _ => panic!("expected named type for param"),
        }

        let ret_ty = func.return_type.as_ref().expect("missing return type");
        match ret_ty {
            TypeExpr::Named { path, .. } => {
                assert_eq!(path.segments, vec!["b", "Foo"]);
                assert!(
                    path.resolved_module.is_some(),
                    "return type should have resolved_module"
                );
            }
            _ => panic!("expected named type for return"),
        }
    }

    #[test]
    fn annotates_paths_with_module_info() {
        let dir = tempfile::tempdir().unwrap();
        let module_main = dir.path().join("main.sio");
        let module_math = dir.path().join("math.sio");

        fs::write(&module_math, "fn sin(x: f64) -> f64 { return x }").unwrap();
        fs::write(
            &module_main,
            "import math;\n\nfn test() -> f64 {\n    let x = math.sin(1.0)\n    return x\n}\n",
        )
        .unwrap();

        let ast = load_program_ast(&module_main).unwrap();

        // Find the test function from main module
        let func = ast.items.iter().find_map(|item| match item {
            Item::Function(f) if f.name == "test" => Some(f),
            _ => None,
        });
        let func = func.expect("missing test() function");

        // Verify the function has a body with statements
        assert!(
            !func.body.stmts.is_empty(),
            "Expected test() to have statements in body"
        );

        // Verify the imported sin function exists inside the math module
        let math_module = ast.items.iter().find_map(|item| match item {
            Item::Module(m) if m.name == "math" => Some(m),
            _ => None,
        });
        assert!(math_module.is_some(), "Expected math module from import");
        let math_items = math_module.unwrap().items.as_ref().unwrap();
        let sin_func = math_items.iter().find_map(|item| match item {
            Item::Function(f) if f.name == "sin" => Some(f),
            _ => None,
        });
        assert!(
            sin_func.is_some(),
            "Expected sin() function inside math module"
        );
    }

    #[test]
    fn module_id_from_file_path() {
        let path = std::path::Path::new("/some/path/mymodule.sio");
        let module_id = ModuleId::from_file_path(path);
        assert_eq!(module_id.path, vec!["mymodule"]);
        assert_eq!(format!("{}", module_id), "mymodule");
    }

    #[test]
    fn module_id_root() {
        let root = ModuleId::root();
        assert!(root.is_root());
        assert_eq!(format!("{}", root), "<root>");
    }

    #[test]
    fn module_id_join() {
        let base = ModuleId::new(vec!["std".to_string()]);
        let joined = base.join("math");
        assert_eq!(joined.path, vec!["std", "math"]);
    }
}
