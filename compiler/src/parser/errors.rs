//! Parser Error Diagnostics
//!
//! Provides context-aware, user-friendly error messages for parser failures.
//! This module detects common patterns that indicate users are trying to use
//! unimplemented or unsupported syntax and provides helpful guidance.

use crate::common::Span;
use crate::lexer::TokenKind;
use miette::{Diagnostic, SourceSpan};
use thiserror::Error;

/// A rich parser error with context and suggestions
#[derive(Debug, Error, Diagnostic)]
pub enum ParserError {
    /// Refinement type syntax attempted but not yet implemented
    #[error("Refinement type syntax not yet implemented")]
    #[diagnostic(
        code(P0010),
        help(
            "The syntax `{{ x: Type | constraint }}` is planned but not yet available.\nConsider using a regular type with runtime validation for now."
        )
    )]
    RefinementTypeNotImplemented {
        #[label("refinement type syntax attempted here")]
        span: SourceSpan,
    },

    /// Rust-style mutable reference attempted
    #[error("Sounio uses `&!` for mutable references, not `&mut`")]
    #[diagnostic(
        code(P0011),
        help(
            "Replace `&mut T` with `&!T` for exclusive/mutable references.\nSounio uses `&T` for shared references and `&!T` for mutable references."
        )
    )]
    RustMutReference {
        #[label("use `&!` instead of `&mut`")]
        span: SourceSpan,
    },

    /// Tuple destructuring attempted
    #[error("Tuple destructuring in patterns is not yet implemented")]
    #[diagnostic(
        code(P0012),
        help("Instead of `let (a, b) = tuple`, use:\n  let a = tuple.0;\n  let b = tuple.1;")
    )]
    TupleDestructuring {
        #[label("tuple destructuring pattern here")]
        span: SourceSpan,
    },

    /// Rust-style macro invocation attempted
    #[error("Rust-style macros are not supported in Sounio")]
    #[diagnostic(
        code(P0013),
        help("Sounio does not support Rust macros like `{}!(...)`.\n{}", .macro_name, .alternative)
    )]
    RustMacroNotSupported {
        #[label("Rust macro syntax not supported")]
        span: SourceSpan,
        macro_name: String,
        alternative: String,
    },

    /// Rust attribute syntax attempted
    #[error("Rust-style attribute syntax is not supported")]
    #[diagnostic(
        code(P0014),
        help("Sounio uses `#[name]` for attributes, but Rust-specific attributes\nlike `#[derive(...)]` and `#[test]` are not available.\n{}", .suggestion)
    )]
    RustAttributeNotSupported {
        #[label("unsupported attribute")]
        span: SourceSpan,
        suggestion: String,
    },

    /// Lambda/closure with tuple destructuring
    #[error("Tuple destructuring in closure parameters is not supported")]
    #[diagnostic(
        code(P0015),
        help(
            "Instead of `|(a, b)| expr`, use:\n  |pair| {{ let a = pair.0; let b = pair.1; expr }}"
        )
    )]
    ClosureTupleDestructuring {
        #[label("closure with tuple destructuring")]
        span: SourceSpan,
    },

    /// Generic expected token error with context
    #[error("Expected {expected}, found `{found}`")]
    #[diagnostic(code(P0001))]
    UnexpectedToken {
        #[label("{context}")]
        span: SourceSpan,
        expected: String,
        found: String,
        context: String,
    },

    /// Expected a type but found something else
    #[error("Expected a type expression")]
    #[diagnostic(code(P0003))]
    ExpectedType {
        #[label("expected a type here")]
        span: SourceSpan,
        #[help]
        help: Option<String>,
    },

    /// Expected an expression but found something else
    #[error("Expected an expression")]
    #[diagnostic(code(P0002))]
    ExpectedExpression {
        #[label("expected an expression here")]
        span: SourceSpan,
        #[help]
        help: Option<String>,
    },

    /// Expected a pattern but found something else
    #[error("Expected a pattern")]
    #[diagnostic(code(P0006))]
    ExpectedPattern {
        #[label("expected a pattern here")]
        span: SourceSpan,
        #[help]
        help: Option<String>,
    },

    /// Missing semicolon
    #[error("Expected `;` after statement")]
    #[diagnostic(code(P0004), help("Add a semicolon `;` at the end of the statement"))]
    MissingSemicolon {
        #[label("expected `;` here")]
        span: SourceSpan,
    },

    /// Mismatched brackets
    #[error("Mismatched {open_kind}")]
    #[diagnostic(code(P0005))]
    MismatchedBracket {
        #[label("this `{open_char}` was never closed")]
        open_span: SourceSpan,
        open_kind: String,
        open_char: char,
        #[label("expected `{expected_close}`, found `{found_close}`")]
        close_span: Option<SourceSpan>,
        expected_close: char,
        found_close: String,
    },

    /// Feature not yet implemented
    #[error("{feature} is not yet implemented")]
    #[diagnostic(code(P0020))]
    FeatureNotImplemented {
        #[label("{feature} used here")]
        span: SourceSpan,
        feature: String,
        #[help]
        help: Option<String>,
    },

    /// Invalid item at module level
    #[error("Invalid item at module level")]
    #[diagnostic(code(P0021))]
    InvalidModuleLevelItem {
        #[label("unexpected item")]
        span: SourceSpan,
        #[help]
        help: Option<String>,
    },
}

impl ParserError {
    /// Create error from span and token info
    pub fn from_span(span: Span) -> SourceSpan {
        SourceSpan::new(span.start.into(), span.len())
    }
}

/// Context for generating better error messages
pub struct ErrorContext<'a> {
    /// Current token
    pub current: TokenKind,
    /// Current token text
    pub current_text: &'a str,
    /// Current span
    pub span: Span,
    /// Lookahead tokens (up to 3)
    pub lookahead: [TokenKind; 3],
    /// Source text (for pattern matching)
    pub source: &'a str,
}

impl<'a> ErrorContext<'a> {
    /// Detect if this looks like a refinement type attempt: { x: Type | ... }
    pub fn looks_like_refinement_type(&self) -> bool {
        // Pattern: LBrace Ident Colon ... Pipe
        if self.current != TokenKind::LBrace {
            return false;
        }

        // Check lookahead pattern
        self.lookahead[0] == TokenKind::Ident && self.lookahead[1] == TokenKind::Colon
    }

    /// Detect if this looks like a Rust &mut reference
    pub fn looks_like_rust_mut_ref(&self) -> bool {
        self.current == TokenKind::Amp && self.lookahead[0] == TokenKind::Mut
    }

    /// Detect if this looks like tuple destructuring: (a, b) = ...
    pub fn looks_like_tuple_destructuring(&self) -> bool {
        self.current == TokenKind::LParen && self.lookahead[0] == TokenKind::Ident
    }

    /// Detect if this looks like a Rust macro: name!(...)
    pub fn looks_like_rust_macro(&self, text: &str) -> bool {
        // Common Rust macros that users might try
        matches!(
            text,
            "assert"
                | "println"
                | "print"
                | "vec"
                | "format"
                | "panic"
                | "debug_assert"
                | "eprintln"
                | "eprint"
                | "dbg"
                | "todo"
                | "unimplemented"
        )
    }

    /// Detect if this looks like a Rust attribute: #[derive(...)] or #[test]
    pub fn looks_like_rust_attribute(&self, attr_name: &str) -> bool {
        matches!(
            attr_name,
            "derive"
                | "test"
                | "cfg"
                | "allow"
                | "warn"
                | "deny"
                | "inline"
                | "repr"
                | "doc"
                | "must_use"
                | "deprecated"
        )
    }
}

/// Generate improved error message when expecting a type
pub fn type_error_for_token(token: TokenKind, span: Span, lookahead: &[TokenKind]) -> ParserError {
    let source_span = ParserError::from_span(span);

    match token {
        // { x: Type | constraint } - refinement type syntax
        TokenKind::LBrace => {
            if !lookahead.is_empty() && lookahead[0] == TokenKind::Ident {
                ParserError::RefinementTypeNotImplemented { span: source_span }
            } else {
                ParserError::ExpectedType {
                    span: source_span,
                    help: Some(
                        "Types cannot start with `{`. Did you mean to use a struct type?"
                            .to_string(),
                    ),
                }
            }
        }

        // &mut T - Rust mutable reference
        TokenKind::Mut => ParserError::RustMutReference { span: source_span },

        // Common mistakes
        TokenKind::Semi => ParserError::ExpectedType {
            span: source_span,
            help: Some(
                "A type annotation is required here. Remove the semicolon or add a type."
                    .to_string(),
            ),
        },

        TokenKind::Eq => ParserError::ExpectedType {
            span: source_span,
            help: Some("Missing type annotation before `=`. Use `: Type` syntax.".to_string()),
        },

        // Operators that can't start types
        TokenKind::Plus | TokenKind::Minus | TokenKind::Slash | TokenKind::Percent => {
            ParserError::ExpectedType {
                span: source_span,
                help: Some(format!(
                    "Operators like `{}` cannot appear at the start of a type expression.",
                    token.as_str()
                )),
            }
        }

        // Default case
        _ => ParserError::ExpectedType {
            span: source_span,
            help: Some(format!(
                "Expected a type name, but found `{}`. Valid types include:\n  \
                     - Primitive types: i32, f64, bool, string\n  \
                     - Generic types: Option<T>, Vec<T>\n  \
                     - Reference types: &T, &!T\n  \
                     - Array types: [T; N], [T]",
                token.as_str()
            )),
        },
    }
}

/// Generate improved error message when expecting an expression
pub fn expr_error_for_token(token: TokenKind, span: Span) -> ParserError {
    let source_span = ParserError::from_span(span);

    match token {
        TokenKind::Semi => ParserError::ExpectedExpression {
            span: source_span,
            help: Some(
                "An expression is required here. Did you forget to write a value?".to_string(),
            ),
        },

        TokenKind::RBrace | TokenKind::RBracket | TokenKind::RParen => {
            ParserError::ExpectedExpression {
                span: source_span,
                help: Some("Expression expected before closing bracket.".to_string()),
            }
        }

        TokenKind::Comma => ParserError::ExpectedExpression {
            span: source_span,
            help: Some("Expression expected before `,`. Did you leave an extra comma?".to_string()),
        },

        _ => ParserError::ExpectedExpression {
            span: source_span,
            help: Some(format!(
                "Expected an expression, but found `{}`.`",
                token.as_str()
            )),
        },
    }
}

/// Generate improved error message when expecting a pattern
pub fn pattern_error_for_token(
    token: TokenKind,
    span: Span,
    lookahead: &[TokenKind],
) -> ParserError {
    let source_span = ParserError::from_span(span);

    match token {
        // (a, b) - tuple destructuring
        TokenKind::LParen => {
            if !lookahead.is_empty() && lookahead[0] == TokenKind::Ident {
                ParserError::TupleDestructuring { span: source_span }
            } else {
                ParserError::ExpectedPattern {
                    span: source_span,
                    help: Some("Grouped patterns with parentheses are not supported.".to_string()),
                }
            }
        }

        _ => ParserError::ExpectedPattern {
            span: source_span,
            help: Some(format!(
                "Expected a pattern (identifier, literal, or struct pattern), found `{}`.",
                token.as_str()
            )),
        },
    }
}

/// Generate error for Rust macro usage
pub fn rust_macro_error(name: &str, span: Span) -> ParserError {
    let source_span = ParserError::from_span(span);

    let alternative = match name {
        "println" | "print" => {
            "Use the IO effect with print functions:\n  fn main() with IO { print(\"message\"); }"
                .to_string()
        }
        "assert" | "debug_assert" => {
            "Use the assert function without `!`:\n  assert(condition, \"message\")".to_string()
        }
        "vec" => "Use array literals:\n  let arr = [1, 2, 3];".to_string(),
        "format" => "Use string interpolation or concatenation.".to_string(),
        "panic" | "todo" | "unimplemented" => {
            "Use the Panic effect:\n  fn foo() with Panic { panic(\"message\"); }".to_string()
        }
        "dbg" => "Use debug printing with the IO effect.".to_string(),
        _ => "Sounio has its own syntax for common operations.".to_string(),
    };

    ParserError::RustMacroNotSupported {
        span: source_span,
        macro_name: name.to_string(),
        alternative,
    }
}

/// Generate error for Rust attribute usage
pub fn rust_attribute_error(name: &str, span: Span) -> ParserError {
    let source_span = ParserError::from_span(span);

    let suggestion = match name {
        "derive" => "Sounio does not have automatic derive. Implement traits manually.".to_string(),
        "test" => "Tests are written in separate test files or use the test runner.".to_string(),
        "cfg" => "Conditional compilation uses different syntax in Sounio.".to_string(),
        "inline" => "The compiler handles inlining automatically.".to_string(),
        "repr" => "Use Sounio's layout attributes for FFI types.".to_string(),
        _ => format!("The `#[{}]` attribute is not available in Sounio.", name),
    };

    ParserError::RustAttributeNotSupported {
        span: source_span,
        suggestion,
    }
}

/// Generate error for unimplemented features
pub fn feature_not_implemented(feature: &str, span: Span, workaround: Option<&str>) -> ParserError {
    ParserError::FeatureNotImplemented {
        span: ParserError::from_span(span),
        feature: feature.to_string(),
        help: workaround.map(|s| s.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_type_error() {
        let span = Span::new(10, 15);
        let err = type_error_for_token(
            TokenKind::LBrace,
            span,
            &[TokenKind::Ident, TokenKind::Colon],
        );

        match err {
            ParserError::RefinementTypeNotImplemented { .. } => {}
            _ => panic!("Expected RefinementTypeNotImplemented error"),
        }
    }

    #[test]
    fn test_rust_mut_ref_error() {
        let span = Span::new(0, 3);
        let err = type_error_for_token(TokenKind::Mut, span, &[]);

        match err {
            ParserError::RustMutReference { .. } => {}
            _ => panic!("Expected RustMutReference error"),
        }
    }

    #[test]
    fn test_rust_macro_alternatives() {
        let span = Span::new(0, 10);
        let err = rust_macro_error("println", span);

        match err {
            ParserError::RustMacroNotSupported { alternative, .. } => {
                assert!(alternative.contains("IO effect"));
            }
            _ => panic!("Expected RustMacroNotSupported error"),
        }
    }
}
