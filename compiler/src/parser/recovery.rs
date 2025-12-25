//! Parser error recovery
//!
//! Implements panic-mode error recovery for resilient parsing.
//! The parser continues after errors to collect multiple diagnostics.

use crate::common::Span;
use crate::lexer::TokenKind;

/// Synchronization points for error recovery
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPoint {
    /// Synchronize at statement boundaries
    Statement,
    /// Synchronize at item boundaries (fn, struct, etc.)
    Item,
    /// Synchronize at expression boundaries
    Expression,
    /// Synchronize at block end
    BlockEnd,
    /// Synchronize at comma (for lists)
    Comma,
}

/// Tokens that typically start a new statement
pub const STATEMENT_STARTERS: &[TokenKind] = &[
    TokenKind::Let,
    TokenKind::Const,
    TokenKind::If,
    TokenKind::While,
    TokenKind::For,
    TokenKind::Loop,
    TokenKind::Return,
    TokenKind::Break,
    TokenKind::Continue,
    TokenKind::Match,
];

/// Tokens that typically start a new item
pub const ITEM_STARTERS: &[TokenKind] = &[
    TokenKind::Fn,
    TokenKind::Struct,
    TokenKind::Enum,
    TokenKind::Type,
    TokenKind::Impl,
    TokenKind::Trait,
    TokenKind::Const,
    TokenKind::Pub,
    TokenKind::Linear,
    TokenKind::Affine,
    TokenKind::Kernel,
    TokenKind::Effect,
    TokenKind::Handler,
    TokenKind::Import,
    TokenKind::Use,
    TokenKind::Extern,
];

/// Tokens that typically end a statement
pub const STATEMENT_ENDERS: &[TokenKind] = &[TokenKind::Semi, TokenKind::RBrace];

/// Parse error with recovery information
#[derive(Debug, Clone)]
pub struct ParseError {
    pub message: String,
    pub span: Span,
    pub expected: Vec<TokenKind>,
    pub found: TokenKind,
    pub notes: Vec<String>,
    pub suggestions: Vec<Suggestion>,
}

impl ParseError {
    pub fn new(message: impl Into<String>, span: Span, found: TokenKind) -> Self {
        Self {
            message: message.into(),
            span,
            expected: Vec::new(),
            found,
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    pub fn expected(mut self, tokens: impl IntoIterator<Item = TokenKind>) -> Self {
        self.expected = tokens.into_iter().collect();
        self
    }

    pub fn note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn suggest(mut self, suggestion: Suggestion) -> Self {
        self.suggestions.push(suggestion);
        self
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)?;
        if !self.expected.is_empty() {
            write!(f, " (expected ")?;
            for (i, tok) in self.expected.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", tok)?;
            }
            write!(f, ", found {})", self.found)?;
        }
        Ok(())
    }
}

impl std::error::Error for ParseError {}

/// A suggestion for fixing a parse error
#[derive(Debug, Clone)]
pub struct Suggestion {
    pub message: String,
    pub span: Span,
    pub replacement: String,
}

impl Suggestion {
    pub fn new(message: impl Into<String>, span: Span, replacement: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span,
            replacement: replacement.into(),
        }
    }

    /// Suggest inserting text at a position
    pub fn insert(message: impl Into<String>, pos: usize, text: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            span: Span::new(pos, pos),
            replacement: text.into(),
        }
    }

    /// Suggest deleting a span
    pub fn delete(message: impl Into<String>, span: Span) -> Self {
        Self {
            message: message.into(),
            span,
            replacement: String::new(),
        }
    }
}

/// Parser state for error recovery
#[derive(Debug, Default)]
pub struct RecoveryState {
    /// Whether we're currently in recovery mode
    pub in_recovery: bool,

    /// Collected errors
    pub errors: Vec<ParseError>,

    /// Maximum errors before giving up
    pub max_errors: usize,

    /// Depth of nested structures (parens, braces, brackets)
    pub nesting_depth: NestingDepth,
}

/// Track nesting depth for recovery
#[derive(Debug, Default, Clone, Copy)]
pub struct NestingDepth {
    pub parens: i32,
    pub braces: i32,
    pub brackets: i32,
}

impl NestingDepth {
    pub fn update(&mut self, kind: TokenKind) {
        match kind {
            TokenKind::LParen => self.parens += 1,
            TokenKind::RParen => self.parens -= 1,
            TokenKind::LBrace => self.braces += 1,
            TokenKind::RBrace => self.braces -= 1,
            TokenKind::LBracket => self.brackets += 1,
            TokenKind::RBracket => self.brackets -= 1,
            _ => {}
        }
    }

    pub fn is_balanced(&self) -> bool {
        self.parens >= 0 && self.braces >= 0 && self.brackets >= 0
    }

    pub fn total(&self) -> i32 {
        self.parens + self.braces + self.brackets
    }
}

impl RecoveryState {
    pub fn new() -> Self {
        Self {
            in_recovery: false,
            errors: Vec::new(),
            max_errors: 100,
            nesting_depth: NestingDepth::default(),
        }
    }

    /// Record an error
    pub fn record_error(&mut self, error: ParseError) {
        self.errors.push(error);
        self.in_recovery = true;
    }

    /// Check if we should stop parsing
    pub fn should_stop(&self) -> bool {
        self.errors.len() >= self.max_errors
    }

    /// Exit recovery mode
    pub fn exit_recovery(&mut self) {
        self.in_recovery = false;
    }

    /// Get all collected errors
    pub fn take_errors(&mut self) -> Vec<ParseError> {
        std::mem::take(&mut self.errors)
    }

    /// Check if any errors occurred
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Get error count
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

/// Helper functions for determining synchronization points
pub fn is_statement_starter(kind: TokenKind) -> bool {
    STATEMENT_STARTERS.contains(&kind)
}

pub fn is_item_starter(kind: TokenKind) -> bool {
    ITEM_STARTERS.contains(&kind)
}

pub fn is_statement_ender(kind: TokenKind) -> bool {
    STATEMENT_ENDERS.contains(&kind)
}

/// Determine if a token can start an expression
pub fn can_start_expression(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Ident
            | TokenKind::IntLit
            | TokenKind::BinLit
            | TokenKind::OctLit
            | TokenKind::HexLit
            | TokenKind::FloatLit
            | TokenKind::StringLit
            | TokenKind::CharLit
            | TokenKind::True
            | TokenKind::False
            | TokenKind::LParen
            | TokenKind::LBracket
            | TokenKind::LBrace
            | TokenKind::If
            | TokenKind::Match
            | TokenKind::Loop
            | TokenKind::While
            | TokenKind::For
            | TokenKind::Return
            | TokenKind::Break
            | TokenKind::Continue
            | TokenKind::Minus
            | TokenKind::Bang
            | TokenKind::Amp
            | TokenKind::Star
            | TokenKind::Pipe
            | TokenKind::SelfLower
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nesting_depth() {
        let mut depth = NestingDepth::default();

        depth.update(TokenKind::LParen);
        assert_eq!(depth.parens, 1);

        depth.update(TokenKind::LBrace);
        assert_eq!(depth.braces, 1);

        depth.update(TokenKind::RParen);
        assert_eq!(depth.parens, 0);

        assert!(depth.is_balanced());
    }

    #[test]
    fn test_recovery_state() {
        let mut state = RecoveryState::new();
        assert!(!state.has_errors());

        state.record_error(ParseError::new(
            "test error",
            Span::new(0, 5),
            TokenKind::Ident,
        ));

        assert!(state.has_errors());
        assert!(state.in_recovery);
        assert_eq!(state.error_count(), 1);
    }

    #[test]
    fn test_sync_helpers() {
        assert!(is_statement_starter(TokenKind::Let));
        assert!(is_statement_starter(TokenKind::If));
        assert!(!is_statement_starter(TokenKind::Plus));

        assert!(is_item_starter(TokenKind::Fn));
        assert!(is_item_starter(TokenKind::Struct));
        assert!(!is_item_starter(TokenKind::Let));

        assert!(can_start_expression(TokenKind::Ident));
        assert!(can_start_expression(TokenKind::IntLit));
        assert!(!can_start_expression(TokenKind::Plus));
    }
}
