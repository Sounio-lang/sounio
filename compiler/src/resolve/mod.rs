//! Name resolution pass
//!
//! Resolves all names in the AST to their definitions.
//! Produces a resolved AST where every name reference has a DefId.

mod resolver;
mod symbols;

pub use resolver::{ResolvedAst, Resolver, resolve};
pub use symbols::{DefId, DefKind, Scope, ScopeKind, Symbol, SymbolTable};
