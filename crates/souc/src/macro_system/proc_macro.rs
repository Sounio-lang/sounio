//! Procedural macro infrastructure

use std::collections::HashMap;
use std::fmt;
use std::iter::FromIterator;

use super::token_tree::*;
use crate::common::Span;

/// A stream of tokens for procedural macros
#[derive(Debug, Clone, Default)]
pub struct TokenStream {
    trees: Vec<TokenTree>,
}

impl TokenStream {
    pub fn new() -> Self {
        TokenStream { trees: Vec::new() }
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    pub fn push(&mut self, tree: TokenTree) {
        self.trees.push(tree);
    }

    pub fn extend(&mut self, other: TokenStream) {
        self.trees.extend(other.trees);
    }

    pub fn into_trees(self) -> Vec<TokenTree> {
        self.trees
    }

    pub fn trees(&self) -> &[TokenTree] {
        &self.trees
    }
}

impl FromIterator<TokenTree> for TokenStream {
    fn from_iter<I: IntoIterator<Item = TokenTree>>(iter: I) -> Self {
        TokenStream {
            trees: iter.into_iter().collect(),
        }
    }
}

impl IntoIterator for TokenStream {
    type Item = TokenTree;
    type IntoIter = std::vec::IntoIter<TokenTree>;

    fn into_iter(self) -> Self::IntoIter {
        self.trees.into_iter()
    }
}

impl fmt::Display for TokenStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for tree in &self.trees {
            write!(f, "{} ", format_tree(tree))?;
        }
        Ok(())
    }
}

fn format_tree(tree: &TokenTree) -> String {
    match tree {
        TokenTree::Token(t) => t.token.text.clone(),
        TokenTree::Delimited(Delimiter::Parenthesis, inner, _) => {
            format!(
                "({})",
                inner.iter().map(format_tree).collect::<Vec<_>>().join(" ")
            )
        }
        TokenTree::Delimited(Delimiter::Bracket, inner, _) => {
            format!(
                "[{}]",
                inner.iter().map(format_tree).collect::<Vec<_>>().join(" ")
            )
        }
        TokenTree::Delimited(Delimiter::Brace, inner, _) => {
            format!(
                "{{{}}}",
                inner.iter().map(format_tree).collect::<Vec<_>>().join(" ")
            )
        }
        TokenTree::Delimited(Delimiter::None, inner, _) => {
            inner.iter().map(format_tree).collect::<Vec<_>>().join(" ")
        }
    }
}

/// A procedural macro definition
#[derive(Debug, Clone)]
pub struct ProcMacroDef {
    pub name: String,
    pub kind: ProcMacroKind,
    pub implementation: ProcMacroImpl,
}

/// Kinds of procedural macros
#[derive(Debug, Clone)]
pub enum ProcMacroKind {
    FunctionLike,
    Derive {
        trait_name: String,
        attributes: Vec<String>,
    },
    Attribute {
        targets: Vec<AttributeTarget>,
    },
}

/// Valid attribute targets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttributeTarget {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Module,
    Field,
    Variant,
    Statement,
    Expression,
}

/// Procedural macro implementation
#[derive(Debug, Clone)]
pub enum ProcMacroImpl {
    Native(NativeProcMacro),
    Interpreted(InterpretedProcMacro),
}

/// Native proc macro (function pointer)
#[derive(Clone)]
pub struct NativeProcMacro {
    pub func: fn(TokenStream) -> Result<TokenStream, ProcMacroError>,
    pub attr_func: Option<fn(TokenStream, TokenStream) -> Result<TokenStream, ProcMacroError>>,
}

impl fmt::Debug for NativeProcMacro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NativeProcMacro").finish()
    }
}

/// Interpreted proc macro
#[derive(Debug, Clone)]
pub struct InterpretedProcMacro {
    pub module_path: String,
    pub function_name: String,
}

/// Proc macro error
#[derive(Debug, Clone)]
pub struct ProcMacroError {
    pub message: String,
    pub span: Option<Span>,
    pub help: Option<String>,
}

impl ProcMacroError {
    pub fn new(message: impl Into<String>) -> Self {
        ProcMacroError {
            message: message.into(),
            span: None,
            help: None,
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_help(mut self, help: impl Into<String>) -> Self {
        self.help = Some(help.into());
        self
    }
}

impl fmt::Display for ProcMacroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)?;
        if let Some(ref help) = self.help {
            write!(f, "\nhelp: {}", help)?;
        }
        Ok(())
    }
}

impl std::error::Error for ProcMacroError {}

/// Procedural macro registry
pub struct ProcMacroRegistry {
    macros: HashMap<String, ProcMacroDef>,
    derives: HashMap<String, String>,
}

impl ProcMacroRegistry {
    pub fn new() -> Self {
        ProcMacroRegistry {
            macros: HashMap::new(),
            derives: HashMap::new(),
        }
    }

    pub fn register(&mut self, macro_def: ProcMacroDef) {
        if let ProcMacroKind::Derive { ref trait_name, .. } = macro_def.kind {
            self.derives
                .insert(trait_name.clone(), macro_def.name.clone());
        }
        self.macros.insert(macro_def.name.clone(), macro_def);
    }

    pub fn get(&self, name: &str) -> Option<&ProcMacroDef> {
        self.macros.get(name)
    }

    pub fn get_derive(&self, trait_name: &str) -> Option<&ProcMacroDef> {
        self.derives
            .get(trait_name)
            .and_then(|name| self.macros.get(name))
    }

    pub fn invoke_function(
        &self,
        name: &str,
        input: TokenStream,
    ) -> Result<TokenStream, ProcMacroError> {
        let macro_def = self
            .get(name)
            .ok_or_else(|| ProcMacroError::new(format!("undefined proc macro: {}", name)))?;

        match &macro_def.implementation {
            ProcMacroImpl::Native(native) => (native.func)(input),
            ProcMacroImpl::Interpreted(_) => {
                Err(ProcMacroError::new("interpreted macros not yet supported"))
            }
        }
    }

    pub fn invoke_attribute(
        &self,
        name: &str,
        attr: TokenStream,
        item: TokenStream,
    ) -> Result<TokenStream, ProcMacroError> {
        let macro_def = self
            .get(name)
            .ok_or_else(|| ProcMacroError::new(format!("undefined attribute macro: {}", name)))?;

        match &macro_def.implementation {
            ProcMacroImpl::Native(native) => {
                let func = native.attr_func.ok_or_else(|| {
                    ProcMacroError::new(format!("{} is not an attribute macro", name))
                })?;
                func(attr, item)
            }
            ProcMacroImpl::Interpreted(_) => {
                Err(ProcMacroError::new("interpreted macros not yet supported"))
            }
        }
    }

    pub fn invoke_derive(
        &self,
        trait_name: &str,
        item: TokenStream,
    ) -> Result<TokenStream, ProcMacroError> {
        let macro_def = self
            .get_derive(trait_name)
            .ok_or_else(|| ProcMacroError::new(format!("no derive macro for: {}", trait_name)))?;

        match &macro_def.implementation {
            ProcMacroImpl::Native(native) => (native.func)(item),
            ProcMacroImpl::Interpreted(_) => {
                Err(ProcMacroError::new("interpreted macros not yet supported"))
            }
        }
    }
}

impl Default for ProcMacroRegistry {
    fn default() -> Self {
        Self::new()
    }
}
