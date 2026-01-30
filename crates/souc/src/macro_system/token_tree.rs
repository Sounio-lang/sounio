//! Token tree representation for macro processing
//!
//! Based on: "Macro-by-Example Revisited" (Kohlbecker et al., 1986)
//! Extended with modern hygiene from "Macros That Work" (Clinger & Rees, 1991)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::common::Span;
use crate::lexer::{Token, TokenKind};

/// Unique syntax context for hygiene
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SyntaxContext(u64);

impl SyntaxContext {
    /// The root context (unhygienic)
    pub const ROOT: Self = SyntaxContext(0);

    /// Generate a fresh context
    pub fn fresh() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        SyntaxContext(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// A set of marks for tracking macro expansion
#[derive(Debug, Clone, Default)]
pub struct MarkSet {
    marks: HashMap<SyntaxContext, Vec<Mark>>,
}

/// A single expansion mark
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mark {
    pub macro_id: u64,
    pub phase: u32,
}

impl MarkSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_mark(&mut self, ctx: SyntaxContext, mark: Mark) {
        self.marks.entry(ctx).or_default().push(mark);
    }

    pub fn is_subset_at(&self, from: SyntaxContext, to: SyntaxContext) -> bool {
        let from_marks = self.marks.get(&from).map(|v| v.as_slice()).unwrap_or(&[]);
        let to_marks = self.marks.get(&to).map(|v| v.as_slice()).unwrap_or(&[]);
        to_marks.iter().all(|m| from_marks.contains(m))
    }
}

/// A token tree node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TokenTree {
    /// A single token
    Token(TokenWithCtx),

    /// A delimited group
    Delimited(Delimiter, Vec<TokenTree>, Span),
}

/// A token with syntax context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenWithCtx {
    pub token: Token,
    pub ctx: SyntaxContext,
}

impl TokenWithCtx {
    pub fn new(token: Token) -> Self {
        TokenWithCtx {
            token,
            ctx: SyntaxContext::ROOT,
        }
    }

    pub fn with_context(token: Token, ctx: SyntaxContext) -> Self {
        TokenWithCtx { token, ctx }
    }
}

/// Delimiter types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Delimiter {
    Parenthesis,
    Bracket,
    Brace,
    None,
}

impl TokenTree {
    pub fn span(&self) -> Span {
        match self {
            TokenTree::Token(t) => t.token.span,
            TokenTree::Delimited(_, _, span) => *span,
        }
    }

    pub fn is_token(&self, kind: TokenKind) -> bool {
        matches!(self, TokenTree::Token(t) if t.token.kind == kind)
    }

    pub fn as_token(&self) -> Option<&TokenWithCtx> {
        match self {
            TokenTree::Token(t) => Some(t),
            _ => None,
        }
    }

    pub fn as_group(&self) -> Option<(Delimiter, &[TokenTree])> {
        match self {
            TokenTree::Delimited(d, trees, _) => Some((*d, trees)),
            _ => None,
        }
    }

    pub fn with_context(self, ctx: SyntaxContext) -> Self {
        match self {
            TokenTree::Token(mut t) => {
                t.ctx = ctx;
                TokenTree::Token(t)
            }
            TokenTree::Delimited(d, trees, span) => {
                let trees = trees.into_iter().map(|t| t.with_context(ctx)).collect();
                TokenTree::Delimited(d, trees, span)
            }
        }
    }
}

/// Macro processing error
#[derive(Debug, Clone)]
pub enum MacroError {
    UnclosedDelimiter {
        span: Span,
    },
    UnexpectedClosingDelimiter {
        span: Span,
    },
    PatternMismatch {
        expected: String,
        found: String,
        span: Span,
    },
    RecursionLimit {
        depth: usize,
        span: Span,
    },
    UndefinedMetaVariable {
        name: String,
        span: Span,
    },
    InvalidRepetition {
        span: Span,
    },
    HygieneViolation {
        name: String,
        span: Span,
    },
}

impl fmt::Display for MacroError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MacroError::UnclosedDelimiter { .. } => write!(f, "unclosed delimiter"),
            MacroError::UnexpectedClosingDelimiter { .. } => {
                write!(f, "unexpected closing delimiter")
            }
            MacroError::PatternMismatch {
                expected, found, ..
            } => {
                write!(
                    f,
                    "pattern mismatch: expected {}, found {}",
                    expected, found
                )
            }
            MacroError::RecursionLimit { depth, .. } => {
                write!(f, "macro recursion limit ({}) exceeded", depth)
            }
            MacroError::UndefinedMetaVariable { name, .. } => {
                write!(f, "undefined metavariable: ${}", name)
            }
            MacroError::InvalidRepetition { .. } => write!(f, "invalid repetition"),
            MacroError::HygieneViolation { name, .. } => {
                write!(f, "hygiene violation: identifier '{}' not in scope", name)
            }
        }
    }
}

impl std::error::Error for MacroError {}
