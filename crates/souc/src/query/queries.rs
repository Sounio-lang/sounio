#![allow(unused_imports)]
//! Concrete query definitions for the compiler

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use super::database::{Durability, QueryDatabase, QueryKey};
use crate::ast::{Ast, Item};
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

/// Query key for HIR (type-checked result)
/// Note: TypedAst doesn't exist - check::check returns Hir directly
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct HirQuery(pub PathBuf);

impl QueryKey for HirQuery {
    type Value = Arc<Hir>;
}

/// Simple function signature for queries (not the full type system)
#[derive(Debug, Clone, Default)]
pub struct SimpleFunctionSignature {
    pub name: String,
    pub params: Vec<String>,
    pub return_type: String,
    pub effects: Vec<String>,
}

/// Query key for function signature
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct FunctionSignatureQuery {
    pub file: PathBuf,
    pub name: String,
}

impl QueryKey for FunctionSignatureQuery {
    type Value = Arc<SimpleFunctionSignature>;
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
    fn resolved_ast(&self, path: PathBuf) -> Option<Arc<ResolvedAst>>;

    /// Get HIR (type-checked)
    fn hir(&self, path: PathBuf) -> Option<Arc<Hir>>;

    /// Get function signature
    fn function_signature(&self, file: PathBuf, name: String) -> Arc<SimpleFunctionSignature>;

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

            // Lex then parse
            match crate::lexer::lex(&contents) {
                Ok(tokens) => match crate::parser::parse(&tokens, &contents) {
                    Ok(ast) => Arc::new(ast),
                    Err(_) => Arc::new(Ast::default()),
                },
                Err(_) => Arc::new(Ast::default()),
            }
        })
    }

    fn resolved_ast(&self, path: PathBuf) -> Option<Arc<ResolvedAst>> {
        self.query(ResolvedAstQuery(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());
            // Clone the AST since resolve takes ownership
            match crate::resolve::resolve((*ast).clone()) {
                Ok(resolved) => Some(Arc::new(resolved)),
                Err(_) => None,
            }
        })
    }

    fn hir(&self, path: PathBuf) -> Option<Arc<Hir>> {
        self.query(HirQuery(path.clone()), |db, key| {
            let resolved = db.resolved_ast(key.0.clone())?;
            // check::check returns Hir directly
            match crate::check::check(&resolved) {
                Ok(hir) => Some(Arc::new(hir)),
                Err(_) => None,
            }
        })
    }

    fn function_signature(&self, file: PathBuf, name: String) -> Arc<SimpleFunctionSignature> {
        self.query(
            FunctionSignatureQuery {
                file: file.clone(),
                name: name.clone(),
            },
            |db, key| {
                let ast = db.parsed_ast(key.file.clone());

                // Find function in AST
                for item in &ast.items {
                    if let Item::Function(f) = item {
                        if f.name == key.name {
                            return Arc::new(SimpleFunctionSignature {
                                name: f.name.clone(),
                                params: f.params.iter().map(|p| format_type_expr(&p.ty)).collect(),
                                return_type: f
                                    .return_type
                                    .as_ref()
                                    .map(|t| format_type_expr(t))
                                    .unwrap_or_else(|| "()".to_string()),
                                effects: f.effects.iter().map(|e| e.name.clone()).collect(),
                            });
                        }
                    }
                }

                Arc::new(SimpleFunctionSignature::default())
            },
        )
    }

    fn module_dependencies(&self, path: PathBuf) -> Arc<Vec<PathBuf>> {
        self.query(ModuleDependencies(path.clone()), |db, key| {
            let ast = db.parsed_ast(key.0.clone());

            let mut deps = Vec::new();
            for item in &ast.items {
                if let Item::Import(import) = item {
                    // Convert import path to file path
                    let dep_path = import.path.to_string().replace("::", "/") + ".sio";
                    deps.push(PathBuf::from(dep_path));
                }
            }

            Arc::new(deps)
        })
    }
}
