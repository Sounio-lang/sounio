//! Name resolution pass
//!
//! Resolves all names in the AST to their definitions.
//! Produces a resolved AST where every name reference has a DefId.
//!
//! ## Multi-pass Import Resolution
//!
//! The module system uses a multi-pass algorithm to resolve imports:
//! 1. First pass: Collect all definitions and build the module tree
//! 2. Import resolution: Iteratively resolve imports until fixpoint
//! 3. Second pass: Resolve all references in function bodies etc.
//!
//! This handles forward references, transitive imports, and detects cycles.

mod resolver;
mod symbols;
pub mod module_tree;

pub use module_tree::ImportError;
pub use resolver::{ResolvedAst, Resolver, resolve};
pub use symbols::{DefId, DefKind, Scope, ScopeKind, Symbol, SymbolTable};
