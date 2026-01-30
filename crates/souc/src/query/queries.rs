#![allow(unused_imports)]
//! Concrete query definitions for the compiler

use std::path::PathBuf;
use std::sync::Arc;

use super::database::{Durability, QueryDatabase, QueryKey};
use crate::ast::Ast;
use crate::check::TypedAst;
use crate::hir::Hir;
use crate::resolve::ResolvedAst;

/// Query key for file contents
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FileContents(pub PathBuf);

impl QueryKey for FileContents {
    type Value = Arc<String>;
}

/// Query key for parsed AST
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ParsedAst(pub PathBuf);

impl QueryKey for ParsedAst {
    type Value = Arc<Ast>;
}

/// Query key for resolved AST
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ResolvedAstQuery(pub PathBuf);

impl QueryKey for ResolvedAstQuery {
    type Value = Arc<ResolvedAst>;
}

/// Query key for type-checked AST
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypeCheckedAst(pub PathBuf);

impl QueryKey for TypeCheckedAst {
    type Value = Arc<TypedAst>;
}

/// Query key for HIR
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HirQuery(pub PathBuf);

impl QueryKey for HirQuery {
    type Value = Arc<Hir>;
}

/// Query key for function signature
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FunctionSignature {
    pub file: PathBuf,
    pub name: String,
}

impl QueryKey for FunctionSignature {
    type Value = Arc<crate::types::FunctionType>;
}

/// Query key for module dependencies
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ModuleDependencies(pub PathBuf);

impl QueryKey for ModuleDependencies {
    type Value = Arc<Vec<PathBuf>>;
}

/// Extension trait for QueryDatabase with compiler queries
pub trait CompilerQueries {
    /// Get file contents (input)
    fn file_contents(&self, path: PathBuf) -> Arc<String>;

    /// Set file contents (input)
    fn set_file_contents(&self, path: PathBuf, contents: String);

    /// Get parsed AST
    fn parsed_ast(&self, path: PathBuf) -> Arc<Ast>;

    /// Get resolved AST
    fn resolved_ast(&self, path: PathBuf) -> Arc<ResolvedAst>;

    /// Get type-checked AST
    fn type_checked_ast(&self, path: PathBuf) -> Arc<TypedAst>;

    /// Get HIR
    fn hir(&self, path: PathBuf) -> Arc<Hir>;

    /// Get function signature
    fn function_signature(&self, file: PathBuf, name: String) -> Arc<crate::types::FunctionType>;

    /// Get module dependencies
    fn module_dependencies(&self, path: PathBuf) -> Arc<Vec<PathBuf>>;
}

impl CompilerQueries for QueryDatabase {
    fn file_contents(&self, path: PathBuf) -> Arc<String> {
        self.query(FileContents(path.clone()), |_db, key| {
            let contents = std::fs::read_to_string(&key.0).unwrap_or_default();
            Arc::new(contents)
        })
    }

    fn set_file_contents(&self, path: PathBuf, contents: String) {
        self.set_input(FileContents(path), Arc::new(contents), Durability::Low);
    }

    fn parsed_ast(&self, path: PathBuf) -> Arc<Ast> {
        self.query(ParsedAst(path.clone()), |db, key| {
            let contents = db.file_contents(key.0.clone());
            let ast = crate::parser::parse(&contents).unwrap_or_default();
            Arc::new(ast)
        })
    }

    fn resolved_ast(&self, path: PathBuf) -> Arc<ResolvedAst> {
        self.query(ResolvedAstQuery(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());
            let resolved = crate::resolve::resolve(&ast).unwrap_or_default();
            Arc::new(resolved)
        })
    }

    fn type_checked_ast(&self, path: PathBuf) -> Arc<TypedAst> {
        self.query(TypeCheckedAst(path.clone()), |db, key| {
            let resolved = db.resolved_ast(key.0.clone());
            let typed = crate::check::type_check(&resolved).unwrap_or_default();
            Arc::new(typed)
        })
    }

    fn hir(&self, path: PathBuf) -> Arc<Hir> {
        self.query(HirQuery(path.clone()), |db, key| {
            let typed = db.type_checked_ast(key.0.clone());
            let hir = crate::hir::lower(&typed).unwrap_or_default();
            Arc::new(hir)
        })
    }

    fn function_signature(&self, file: PathBuf, name: String) -> Arc<crate::types::FunctionType> {
        self.query(
            FunctionSignature {
                file: file.clone(),
                name: name.clone(),
            },
            |db, key| {
                let typed = db.type_checked_ast(key.file.clone());

                // Find function in typed AST
                for item in &typed.items {
                    if let crate::ast::ItemKind::Function(f) = &item.kind {
                        if f.name == key.name {
                            return Arc::new(f.signature.clone());
                        }
                    }
                }

                Arc::new(crate::types::FunctionType::default())
            },
        )
    }

    fn module_dependencies(&self, path: PathBuf) -> Arc<Vec<PathBuf>> {
        self.query(ModuleDependencies(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());

            let mut deps = Vec::new();
            for item in &ast.items {
                if let crate::ast::ItemKind::Use(u) = &item.kind {
                    // Convert use path to file path
                    let dep_path = u.path.replace("::", "/") + ".sio";
                    deps.push(PathBuf::from(dep_path));
                }
            }

            Arc::new(deps)
        })
    }
}

