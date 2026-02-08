//! Macro system and compile-time metaprogramming
//!
//! Implements:
//! - Declarative macros (macro_rules!)
//! - Procedural macros (derive, attribute, function-like)
//! - Compile-time function execution (CTFE)
//! - Scientific domain-specific macros

pub mod ctfe;
pub mod declarative;
pub mod derive;
pub mod pattern;
pub mod proc_macro;
pub mod scientific;
pub mod token_tree;

pub use ctfe::{ConstValue, CtfeContext, CtfeError};
pub use declarative::{MacroArm, MacroDef, MacroExpander};
pub use derive::{DeriveInput, parse_derive_input};
pub use pattern::{Bindings, FragmentSpecifier, Pattern, PatternMatcher};
pub use proc_macro::{ProcMacroDef, ProcMacroError, ProcMacroKind, ProcMacroRegistry, TokenStream};
pub use token_tree::{Delimiter, MacroError, SyntaxContext, TokenTree, TokenWithCtx};

/// Macro expansion context
pub struct MacroContext {
    /// Declarative macro expander
    pub declarative: MacroExpander,

    /// Procedural macro registry
    pub proc_macros: ProcMacroRegistry,

    /// Compile-time evaluation context
    pub ctfe: CtfeContext,
}

impl MacroContext {
    pub fn new() -> Self {
        MacroContext {
            declarative: MacroExpander::new(),
            proc_macros: ProcMacroRegistry::new(),
            ctfe: CtfeContext::new(),
        }
    }
}

impl Default for MacroContext {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_macro_context_creation() {
        let ctx = MacroContext::new();
        assert!(ctx.declarative.macros.is_empty());
    }
}
