//! Tree-walking interpreter for HIR
//!
//! Executes HIR directly for rapid semantic testing.

pub mod autodiff;
pub mod builtins;
pub mod causal;
pub mod closure;
pub mod effect_dispatch;
pub mod env;
pub mod eval;
pub mod symbolic;
pub mod value;

pub use autodiff::Tape;
pub use builtins::BuiltinRegistry;
pub use causal::{CausalDAG, CausalModel};
pub use closure::{DCallable, InterpreterClosure, extract_closure};
pub use effect_dispatch::{EffectContext, EffectError, EffectHandler, EffectKind, HandlerState};
pub use env::Environment;
pub use eval::Interpreter;
pub use symbolic::Expr as SymbolicExpr;
pub use value::Value;
