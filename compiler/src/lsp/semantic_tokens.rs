//! Semantic tokens provider for syntax highlighting
//!
//! Provides rich semantic tokens for enhanced syntax highlighting in IDEs.
//! This module supports both lexer-based tokenization (fast, no semantic info)
//! and semantic analysis-based tokenization (slower, full type info).
//!
//! # Supported Token Types
//!
//! Standard LSP tokens plus D-specific extensions:
//! - `effect` - Effect names (IO, Mut, Alloc, etc.)
//! - `unit` - Units of measure (mg, mL, kg, etc.)
//! - `refinement` - Refinement type predicates
//! - `lifetime` - Lifetime annotations
//! - `label` - Block labels
//!
//! # Custom Modifiers
//!
//! - `mutable` - Mutable bindings
//! - `linear` - Linear types (must be used exactly once)
//! - `affine` - Affine types (used at most once)
//! - `pure` - Pure functions (no effects)
//! - `unsafe` - Unsafe blocks/functions

use tower_lsp::lsp_types::*;

use crate::ast::{Ast, Expr, FnDef, Item, Pattern, TypeExpr};
use crate::common::Span;
use crate::lexer::{self, TokenKind};
use crate::resolve::SymbolTable;

// ============================================================================
// Token Type Constants
// ============================================================================

/// Standard LSP token types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum SemanticTokenType {
    Namespace = 0,
    Type = 1,
    Class = 2,
    Enum = 3,
    Interface = 4,
    Struct = 5,
    TypeParameter = 6,
    Parameter = 7,
    Variable = 8,
    Property = 9,
    EnumMember = 10,
    Event = 11,
    Function = 12,
    Method = 13,
    Macro = 14,
    Keyword = 15,
    Modifier = 16,
    Comment = 17,
    String = 18,
    Number = 19,
    Regexp = 20,
    Operator = 21,
    Decorator = 22,
    // D-specific token types
    Effect = 23,
    Unit = 24,
    Refinement = 25,
    Lifetime = 26,
    Label = 27,
    Attribute = 28,
}

impl SemanticTokenType {
    /// Get the LSP token type name
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Namespace => "namespace",
            Self::Type => "type",
            Self::Class => "class",
            Self::Enum => "enum",
            Self::Interface => "interface",
            Self::Struct => "struct",
            Self::TypeParameter => "typeParameter",
            Self::Parameter => "parameter",
            Self::Variable => "variable",
            Self::Property => "property",
            Self::EnumMember => "enumMember",
            Self::Event => "event",
            Self::Function => "function",
            Self::Method => "method",
            Self::Macro => "macro",
            Self::Keyword => "keyword",
            Self::Modifier => "modifier",
            Self::Comment => "comment",
            Self::String => "string",
            Self::Number => "number",
            Self::Regexp => "regexp",
            Self::Operator => "operator",
            Self::Decorator => "decorator",
            Self::Effect => "effect",
            Self::Unit => "unit",
            Self::Refinement => "refinement",
            Self::Lifetime => "lifetime",
            Self::Label => "label",
            Self::Attribute => "attribute",
        }
    }
}

// ============================================================================
// Token Modifier Constants
// ============================================================================

/// Semantic token modifiers (bit flags)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SemanticTokenModifiers(u32);

impl SemanticTokenModifiers {
    pub const NONE: Self = Self(0);
    pub const DECLARATION: Self = Self(1 << 0);
    pub const DEFINITION: Self = Self(1 << 1);
    pub const READONLY: Self = Self(1 << 2);
    pub const STATIC: Self = Self(1 << 3);
    pub const DEPRECATED: Self = Self(1 << 4);
    pub const ABSTRACT: Self = Self(1 << 5);
    pub const ASYNC: Self = Self(1 << 6);
    pub const MODIFICATION: Self = Self(1 << 7);
    pub const DOCUMENTATION: Self = Self(1 << 8);
    pub const DEFAULT_LIBRARY: Self = Self(1 << 9);
    // D-specific modifiers
    pub const MUTABLE: Self = Self(1 << 10);
    pub const LINEAR: Self = Self(1 << 11);
    pub const AFFINE: Self = Self(1 << 12);
    pub const PURE: Self = Self(1 << 13);
    pub const UNSAFE: Self = Self(1 << 14);
    pub const PUB: Self = Self(1 << 15);

    /// Combine modifiers
    pub fn combine(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Get the raw bit flags
    pub fn bits(self) -> u32 {
        self.0
    }
}

impl std::ops::BitOr for SemanticTokenModifiers {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

// ============================================================================
// Semantic Token Builder
// ============================================================================

/// A single semantic token before delta encoding
#[derive(Debug, Clone)]
struct RawSemanticToken {
    /// Byte offset in source
    start: usize,
    /// Length in bytes
    length: usize,
    /// Token type
    token_type: SemanticTokenType,
    /// Token modifiers
    modifiers: SemanticTokenModifiers,
}

/// Builder for constructing semantic tokens with delta encoding
pub struct SemanticTokensBuilder {
    /// Source text for line/column calculation
    source: String,
    /// Collected tokens (unsorted)
    tokens: Vec<RawSemanticToken>,
    /// Line start offsets cache
    line_starts: Vec<usize>,
}

impl SemanticTokensBuilder {
    /// Create a new builder for the given source
    pub fn new(source: &str) -> Self {
        let line_starts = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        Self {
            source: source.to_string(),
            tokens: Vec::new(),
            line_starts,
        }
    }

    /// Add a semantic token
    pub fn push(
        &mut self,
        span: &Span,
        token_type: SemanticTokenType,
        modifiers: SemanticTokenModifiers,
    ) {
        if span.start < span.end && span.end <= self.source.len() {
            self.tokens.push(RawSemanticToken {
                start: span.start,
                length: span.end - span.start,
                token_type,
                modifiers,
            });
        }
    }

    /// Add a token with a custom range
    pub fn push_range(
        &mut self,
        start: usize,
        end: usize,
        token_type: SemanticTokenType,
        modifiers: SemanticTokenModifiers,
    ) {
        if start < end && end <= self.source.len() {
            self.tokens.push(RawSemanticToken {
                start,
                length: end - start,
                token_type,
                modifiers,
            });
        }
    }

    /// Convert byte offset to (line, column) - 0-indexed
    fn offset_to_line_col(&self, offset: usize) -> (u32, u32) {
        let offset = offset.min(self.source.len());

        // Binary search for the line
        let line = match self.line_starts.binary_search(&offset) {
            Ok(i) => i,
            Err(i) => i.saturating_sub(1),
        };

        let line_start = self.line_starts[line];
        let col = offset - line_start;

        (line as u32, col as u32)
    }

    /// Build the final SemanticTokens with delta encoding
    pub fn build(mut self) -> SemanticTokens {
        // Sort tokens by position
        self.tokens.sort_by_key(|t| t.start);

        let mut data = Vec::with_capacity(self.tokens.len());
        let mut prev_line = 0u32;
        let mut prev_col = 0u32;

        for token in &self.tokens {
            let (line, col) = self.offset_to_line_col(token.start);

            // Delta encoding
            let delta_line = line - prev_line;
            let delta_col = if delta_line == 0 { col - prev_col } else { col };

            data.push(SemanticToken {
                delta_line,
                delta_start: delta_col,
                length: token.length as u32,
                token_type: token.token_type as u32,
                token_modifiers_bitset: token.modifiers.bits(),
            });

            prev_line = line;
            prev_col = col;
        }

        SemanticTokens {
            result_id: None,
            data,
        }
    }
}

// ============================================================================
// Semantic Tokens Provider
// ============================================================================

/// Provider for semantic tokens
pub struct SemanticTokensProvider {
    /// Known effect names for highlighting
    known_effects: std::collections::HashSet<&'static str>,
    /// Known unit names
    known_units: std::collections::HashSet<&'static str>,
}

impl SemanticTokensProvider {
    /// Create a new semantic tokens provider
    pub fn new() -> Self {
        let known_effects: std::collections::HashSet<&'static str> = [
            "IO", "Mut", "Alloc", "Panic", "Async", "GPU", "Prob", "Div", "Net", "State", "Exn",
            "Console", "Time", "Random", "File", "Send", "Recv", "Log",
        ]
        .into_iter()
        .collect();

        let known_units: std::collections::HashSet<&'static str> = [
            "m", "km", "cm", "mm", "um", "nm", // length
            "s", "ms", "us", "ns", "min", "hr", // time
            "kg", "g", "mg", "ug", // mass
            "L", "mL", "uL", // volume
            "mol", "mmol", // amount
            "K", "C", "F", // temperature
            "rad", "deg", // angle
            "Hz", "kHz", "MHz", "GHz", // frequency
            "J", "kJ", "eV", // energy
            "W", "kW", "MW", // power
            "Pa", "kPa", "MPa", "bar", "atm", // pressure
            "V", "mV", "kV", // voltage
            "A", "mA", "uA", // current
            "N", "kN", // force
        ]
        .into_iter()
        .collect();

        Self {
            known_effects,
            known_units,
        }
    }

    /// Tokenize source for semantic highlighting (lexer-based, fast)
    pub fn tokenize(&self, source: &str) -> SemanticTokens {
        let mut builder = SemanticTokensBuilder::new(source);

        // Lex the source
        if let Ok(tokens) = lexer::lex(source) {
            for token in &tokens {
                if let Some((token_type, modifiers)) = self.classify_token(&token.kind) {
                    builder.push(&token.span, token_type, modifiers);
                }
            }
        }

        builder.build()
    }

    /// Tokenize with full semantic analysis (slower, more accurate)
    pub fn tokenize_semantic(
        &self,
        source: &str,
        ast: Option<&Ast>,
        symbols: Option<&SymbolTable>,
    ) -> SemanticTokens {
        let mut builder = SemanticTokensBuilder::new(source);

        // First pass: lexer-based tokens for keywords, literals, operators
        if let Ok(tokens) = lexer::lex(source) {
            for token in &tokens {
                if let Some((token_type, modifiers)) = self.classify_token(&token.kind) {
                    // Skip identifiers - we'll classify them semantically
                    if token.kind != TokenKind::Ident {
                        builder.push(&token.span, token_type, modifiers);
                    }
                }
            }
        }

        // Second pass: semantic analysis for identifiers
        if let Some(ast) = ast {
            self.analyze_ast(ast, symbols, &mut builder);
        }

        builder.build()
    }

    /// Classify a lexer token to semantic token type and modifiers
    fn classify_token(
        &self,
        kind: &TokenKind,
    ) -> Option<(SemanticTokenType, SemanticTokenModifiers)> {
        match kind {
            // Keywords
            TokenKind::Fn
            | TokenKind::Let
            | TokenKind::Mut
            | TokenKind::Const
            | TokenKind::If
            | TokenKind::Else
            | TokenKind::While
            | TokenKind::For
            | TokenKind::Loop
            | TokenKind::Match
            | TokenKind::Return
            | TokenKind::Break
            | TokenKind::Continue
            | TokenKind::In
            | TokenKind::As
            | TokenKind::Where
            | TokenKind::Pub
            | TokenKind::SelfLower
            | TokenKind::SelfUpper => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE))
            }

            // Type keywords
            TokenKind::Struct | TokenKind::Enum | TokenKind::Trait | TokenKind::Type => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE))
            }

            // D-specific keywords
            TokenKind::Effect
            | TokenKind::Handler
            | TokenKind::Handle
            | TokenKind::With
            | TokenKind::Perform
            | TokenKind::Resume => Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE)),

            // Linearity keywords - use Modifier type
            TokenKind::Linear => {
                Some((SemanticTokenType::Modifier, SemanticTokenModifiers::LINEAR))
            }
            TokenKind::Affine => {
                Some((SemanticTokenType::Modifier, SemanticTokenModifiers::AFFINE))
            }

            // GPU/async keywords
            TokenKind::Kernel | TokenKind::Device | TokenKind::Shared | TokenKind::Gpu => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::ASYNC))
            }

            TokenKind::Async | TokenKind::Await | TokenKind::Spawn => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::ASYNC))
            }

            // Probabilistic keywords
            TokenKind::Sample | TokenKind::Observe | TokenKind::Infer => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE))
            }

            // Verification keywords
            TokenKind::Proof
            | TokenKind::Invariant
            | TokenKind::Requires
            | TokenKind::Ensures
            | TokenKind::Assert
            | TokenKind::Assume => Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE)),

            // Unsafe/extern
            TokenKind::Unsafe => Some((SemanticTokenType::Keyword, SemanticTokenModifiers::UNSAFE)),
            TokenKind::Extern => Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE)),

            TokenKind::Impl | TokenKind::Move | TokenKind::Copy | TokenKind::Drop => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE))
            }

            // Boolean literals
            TokenKind::True | TokenKind::False => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE))
            }

            // Numeric literals
            TokenKind::IntLit | TokenKind::HexLit | TokenKind::BinLit | TokenKind::OctLit => {
                Some((SemanticTokenType::Number, SemanticTokenModifiers::NONE))
            }
            TokenKind::FloatLit => Some((SemanticTokenType::Number, SemanticTokenModifiers::NONE)),

            // Unit literals - special highlighting
            TokenKind::IntUnitLit | TokenKind::FloatUnitLit => {
                Some((SemanticTokenType::Unit, SemanticTokenModifiers::NONE))
            }

            // String literals
            TokenKind::StringLit | TokenKind::CharLit => {
                Some((SemanticTokenType::String, SemanticTokenModifiers::NONE))
            }

            // Identifiers - would need semantic analysis for proper classification
            TokenKind::Ident => Some((SemanticTokenType::Variable, SemanticTokenModifiers::NONE)),

            // Operators
            TokenKind::Plus
            | TokenKind::Minus
            | TokenKind::Star
            | TokenKind::Slash
            | TokenKind::Percent
            | TokenKind::Caret
            | TokenKind::Amp
            | TokenKind::Pipe
            | TokenKind::Tilde
            | TokenKind::Bang
            | TokenKind::Eq
            | TokenKind::Lt
            | TokenKind::Gt
            | TokenKind::EqEq
            | TokenKind::Ne
            | TokenKind::Le
            | TokenKind::Ge
            | TokenKind::AmpAmp
            | TokenKind::PipePipe
            | TokenKind::Shl
            | TokenKind::Shr
            | TokenKind::PlusEq
            | TokenKind::MinusEq
            | TokenKind::StarEq
            | TokenKind::SlashEq
            | TokenKind::PercentEq
            | TokenKind::AmpEq
            | TokenKind::PipeEq
            | TokenKind::CaretEq
            | TokenKind::ShlEq
            | TokenKind::ShrEq
            | TokenKind::Arrow
            | TokenKind::FatArrow
            | TokenKind::LeftArrow => {
                Some((SemanticTokenType::Operator, SemanticTokenModifiers::NONE))
            }

            // Module/Import keywords
            TokenKind::Module | TokenKind::Import | TokenKind::Export => {
                Some((SemanticTokenType::Keyword, SemanticTokenModifiers::NONE))
            }

            // Skip punctuation and delimiters
            TokenKind::LParen
            | TokenKind::RParen
            | TokenKind::LBracket
            | TokenKind::RBracket
            | TokenKind::LBrace
            | TokenKind::RBrace
            | TokenKind::Comma
            | TokenKind::Semi
            | TokenKind::Colon
            | TokenKind::ColonColon
            | TokenKind::Dot
            | TokenKind::DotDot
            | TokenKind::DotDotDot
            | TokenKind::DotDotEq
            | TokenKind::Question
            | TokenKind::Underscore
            | TokenKind::Eof => None,

            // Attributes
            TokenKind::At | TokenKind::Hash => {
                Some((SemanticTokenType::Decorator, SemanticTokenModifiers::NONE))
            }

            // Special - dollar for template interpolation
            TokenKind::Dollar => Some((SemanticTokenType::Operator, SemanticTokenModifiers::NONE)),

            // Static keyword
            TokenKind::Static => Some((SemanticTokenType::Keyword, SemanticTokenModifiers::STATIC)),

            // Documentation comments
            TokenKind::DocCommentOuter
            | TokenKind::DocCommentInner
            | TokenKind::DocBlockOuter
            | TokenKind::DocBlockInner => Some((
                SemanticTokenType::Comment,
                SemanticTokenModifiers::DOCUMENTATION,
            )),
        }
    }

    /// Analyze AST for semantic tokens
    fn analyze_ast(
        &self,
        ast: &Ast,
        symbols: Option<&SymbolTable>,
        builder: &mut SemanticTokensBuilder,
    ) {
        for item in &ast.items {
            self.analyze_item(item, symbols, builder);
        }
    }

    /// Analyze a top-level item
    fn analyze_item(
        &self,
        item: &Item,
        symbols: Option<&SymbolTable>,
        builder: &mut SemanticTokensBuilder,
    ) {
        match item {
            Item::Function(f) => self.analyze_function(f, symbols, builder),
            Item::Struct(s) => {
                // Struct name
                builder.push(
                    &s.span,
                    SemanticTokenType::Struct,
                    SemanticTokenModifiers::DEFINITION,
                );
            }
            Item::Enum(e) => {
                // Enum name
                builder.push(
                    &e.span,
                    SemanticTokenType::Enum,
                    SemanticTokenModifiers::DEFINITION,
                );
            }
            Item::Trait(t) => {
                // Trait name
                builder.push(
                    &t.span,
                    SemanticTokenType::Interface,
                    SemanticTokenModifiers::DEFINITION,
                );
            }
            Item::Effect(e) => {
                // Effect name
                builder.push(
                    &e.span,
                    SemanticTokenType::Effect,
                    SemanticTokenModifiers::DEFINITION,
                );
            }
            Item::TypeAlias(t) => {
                // Type alias name
                builder.push(
                    &t.span,
                    SemanticTokenType::Type,
                    SemanticTokenModifiers::DEFINITION,
                );
            }
            Item::Global(g) => {
                // Global variable
                builder.push(
                    &g.span,
                    SemanticTokenType::Variable,
                    SemanticTokenModifiers::STATIC | SemanticTokenModifiers::DEFINITION,
                );
            }
            Item::Impl(i) => {
                // Analyze methods
                for impl_item in &i.items {
                    if let crate::ast::ImplItem::Fn(f) = impl_item {
                        self.analyze_function(f, symbols, builder);
                    }
                }
            }
            Item::Import(_) | Item::Handler(_) | Item::Extern(_) => {}
            Item::MacroInvocation(_) => {
                // Macro invocations are expanded before semantic token processing
            }
        }
    }

    /// Analyze a function definition
    fn analyze_function(
        &self,
        f: &FnDef,
        symbols: Option<&SymbolTable>,
        builder: &mut SemanticTokensBuilder,
    ) {
        // Function name
        let mut modifiers = SemanticTokenModifiers::DEFINITION;
        if matches!(f.visibility, crate::ast::Visibility::Public) {
            modifiers = modifiers | SemanticTokenModifiers::PUB;
        }
        if f.effects.is_empty() {
            modifiers = modifiers | SemanticTokenModifiers::PURE;
        }
        if f.modifiers.is_unsafe {
            modifiers = modifiers | SemanticTokenModifiers::UNSAFE;
        }
        if f.modifiers.is_async {
            modifiers = modifiers | SemanticTokenModifiers::ASYNC;
        }

        // The function name span would be more precise, but we use the full span for now
        builder.push(&f.span, SemanticTokenType::Function, modifiers);

        // Parameters
        for param in &f.params {
            self.analyze_pattern(&param.pattern, true, builder);
            self.analyze_type(&param.ty, builder);
        }

        // Return type
        if let Some(ref ret) = f.return_type {
            self.analyze_type(ret, builder);
        }

        // Effects - EffectRef doesn't have span field, skip for now
        // for effect in &f.effects {
        //     // Effect names would need proper span tracking
        // }

        // Body is now a Block, not Option<Expr>
        self.analyze_block(&f.body, symbols, builder);
    }

    /// Analyze a block
    fn analyze_block(
        &self,
        block: &crate::ast::Block,
        symbols: Option<&SymbolTable>,
        builder: &mut SemanticTokensBuilder,
    ) {
        for stmt in &block.stmts {
            self.analyze_stmt(stmt, symbols, builder);
        }
    }

    /// Analyze a pattern
    fn analyze_pattern(
        &self,
        pattern: &Pattern,
        _is_parameter: bool,
        builder: &mut SemanticTokensBuilder,
    ) {
        match pattern {
            Pattern::Binding {
                mutable: _,
                name: _,
            } => {
                // Pattern doesn't have span info, skip for now
                // Would need to add span tracking to Pattern enum
            }
            Pattern::Tuple(elements) => {
                for elem in elements {
                    self.analyze_pattern(elem, _is_parameter, builder);
                }
            }
            Pattern::Struct { path: _, fields } => {
                for (_, field_pat) in fields {
                    self.analyze_pattern(field_pat, _is_parameter, builder);
                }
            }
            Pattern::Enum { path: _, patterns } => {
                if let Some(pats) = patterns {
                    for p in pats {
                        self.analyze_pattern(p, _is_parameter, builder);
                    }
                }
            }
            Pattern::Literal(_) => {
                // Literal patterns - handled by lexer pass
            }
            Pattern::Wildcard | Pattern::Or(_) => {}
        }
        // Suppress unused warning
        let _ = builder;
    }

    /// Analyze a type expression
    fn analyze_type(&self, ty: &TypeExpr, builder: &mut SemanticTokensBuilder) {
        match ty {
            TypeExpr::Named {
                path,
                args,
                unit: _,
            } => {
                // Check if this is a known effect
                let name = path.to_string();
                if self.known_effects.contains(name.as_str()) {
                    // Effect reference - would need span info from path
                } else if self.known_units.contains(name.as_str()) {
                    // Unit reference - would need span info from path
                }
                // Type arguments
                for arg in args {
                    self.analyze_type(arg, builder);
                }
            }
            TypeExpr::Reference { inner, .. } => {
                self.analyze_type(inner, builder);
            }
            TypeExpr::Array { element, .. } => {
                self.analyze_type(element, builder);
            }
            TypeExpr::Tuple(types) => {
                for t in types {
                    self.analyze_type(t, builder);
                }
            }
            TypeExpr::Function {
                params,
                return_type,
                ..
            } => {
                for p in params {
                    self.analyze_type(p, builder);
                }
                self.analyze_type(return_type, builder);
            }
            TypeExpr::Unit | TypeExpr::SelfType | TypeExpr::Infer => {}
        }
        // Suppress unused warning
        let _ = builder;
    }

    /// Analyze an expression
    fn analyze_expr(
        &self,
        expr: &Expr,
        symbols: Option<&SymbolTable>,
        builder: &mut SemanticTokensBuilder,
    ) {
        match expr {
            Expr::Literal { .. } => {
                // Literals are handled by lexer pass
            }
            Expr::Path { path, .. } => {
                // Check if path refers to a known type/effect
                if let Some(name) = path.name() {
                    let (_token_type, _modifiers) = self.classify_identifier(name, symbols);
                    // Would need span info from path to push token
                }
            }
            Expr::Binary { left, right, .. } => {
                self.analyze_expr(left, symbols, builder);
                self.analyze_expr(right, symbols, builder);
            }
            Expr::Unary { expr: inner, .. } => {
                self.analyze_expr(inner, symbols, builder);
            }
            Expr::Call { callee, args, .. } => {
                self.analyze_expr(callee, symbols, builder);
                for arg in args {
                    self.analyze_expr(arg, symbols, builder);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.analyze_expr(receiver, symbols, builder);
                for arg in args {
                    self.analyze_expr(arg, symbols, builder);
                }
            }
            Expr::Field { base, .. } | Expr::TupleField { base, .. } => {
                self.analyze_expr(base, symbols, builder);
            }
            Expr::Index { base, index, .. } => {
                self.analyze_expr(base, symbols, builder);
                self.analyze_expr(index, symbols, builder);
            }
            Expr::Cast {
                expr: inner, ty, ..
            } => {
                self.analyze_expr(inner, symbols, builder);
                self.analyze_type(ty, builder);
            }
            Expr::Block { block, .. } => {
                self.analyze_block(block, symbols, builder);
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
                ..
            } => {
                self.analyze_expr(condition, symbols, builder);
                self.analyze_block(then_branch, symbols, builder);
                if let Some(e) = else_branch {
                    self.analyze_expr(e, symbols, builder);
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                self.analyze_expr(scrutinee, symbols, builder);
                for arm in arms {
                    self.analyze_pattern(&arm.pattern, false, builder);
                    if let Some(guard) = &arm.guard {
                        self.analyze_expr(guard, symbols, builder);
                    }
                    self.analyze_expr(&arm.body, symbols, builder);
                }
            }
            Expr::Loop { body, .. } => {
                self.analyze_block(body, symbols, builder);
            }
            Expr::While {
                condition, body, ..
            } => {
                self.analyze_expr(condition, symbols, builder);
                self.analyze_block(body, symbols, builder);
            }
            Expr::For {
                pattern,
                iter,
                body,
                ..
            } => {
                self.analyze_pattern(pattern, false, builder);
                self.analyze_expr(iter, symbols, builder);
                self.analyze_block(body, symbols, builder);
            }
            Expr::Return { value, .. } | Expr::Break { value, .. } => {
                if let Some(v) = value {
                    self.analyze_expr(v, symbols, builder);
                }
            }
            Expr::Continue { .. } => {}
            Expr::Closure { body, .. } | Expr::AsyncClosure { body, .. } => {
                self.analyze_expr(body, symbols, builder);
            }
            Expr::Tuple { elements, .. } | Expr::Array { elements, .. } => {
                for elem in elements {
                    self.analyze_expr(elem, symbols, builder);
                }
            }
            Expr::StructLit { fields, .. } => {
                for (_, value) in fields {
                    self.analyze_expr(value, symbols, builder);
                }
            }
            Expr::Try { expr: inner, .. }
            | Expr::Await { expr: inner, .. }
            | Expr::Sample {
                distribution: inner,
                ..
            }
            | Expr::Spawn { expr: inner, .. } => {
                self.analyze_expr(inner, symbols, builder);
            }
            Expr::Perform { args, .. } => {
                for arg in args {
                    self.analyze_expr(arg, symbols, builder);
                }
            }
            Expr::Handle { expr: inner, .. } => {
                self.analyze_expr(inner, symbols, builder);
            }
            Expr::AsyncBlock { block, .. } => {
                self.analyze_block(block, symbols, builder);
            }
            Expr::Select { arms, .. } => {
                for arm in arms {
                    self.analyze_expr(&arm.future, symbols, builder);
                    self.analyze_pattern(&arm.pattern, false, builder);
                    if let Some(guard) = &arm.guard {
                        self.analyze_expr(guard, symbols, builder);
                    }
                    self.analyze_expr(&arm.body, symbols, builder);
                }
            }
            Expr::Join { futures, .. } => {
                for f in futures {
                    self.analyze_expr(f, symbols, builder);
                }
            }
            Expr::MacroInvocation(_) => {
                // Macro invocations are expanded before semantic token processing
            }
        }
    }

    /// Analyze a statement
    fn analyze_stmt(
        &self,
        stmt: &crate::ast::Stmt,
        symbols: Option<&SymbolTable>,
        builder: &mut SemanticTokensBuilder,
    ) {
        match stmt {
            crate::ast::Stmt::Let {
                pattern, ty, value, ..
            } => {
                self.analyze_pattern(pattern, false, builder);
                if let Some(t) = ty {
                    self.analyze_type(t, builder);
                }
                if let Some(e) = value {
                    self.analyze_expr(e, symbols, builder);
                }
            }
            crate::ast::Stmt::Expr { expr, .. } => {
                self.analyze_expr(expr, symbols, builder);
            }
            crate::ast::Stmt::Assign { target, value, .. } => {
                self.analyze_expr(target, symbols, builder);
                self.analyze_expr(value, symbols, builder);
            }
            crate::ast::Stmt::Empty => {}
            crate::ast::Stmt::MacroInvocation(_) => {
                // Macro invocations are expanded before semantic token processing
            }
        }
    }

    /// Classify an identifier based on symbol table lookup
    fn classify_identifier(
        &self,
        name: &str,
        _symbols: Option<&SymbolTable>,
    ) -> (SemanticTokenType, SemanticTokenModifiers) {
        // Check for known effects
        if self.known_effects.contains(name) {
            return (SemanticTokenType::Effect, SemanticTokenModifiers::NONE);
        }

        // Check for known units
        if self.known_units.contains(name) {
            return (SemanticTokenType::Unit, SemanticTokenModifiers::NONE);
        }

        // Check naming conventions
        if self.is_type_name(name) {
            return (SemanticTokenType::Type, SemanticTokenModifiers::NONE);
        }

        if name.starts_with('_') {
            return (SemanticTokenType::Variable, SemanticTokenModifiers::NONE);
        }

        // Default to variable
        (SemanticTokenType::Variable, SemanticTokenModifiers::NONE)
    }

    /// Check if a name looks like a type (PascalCase)
    fn is_type_name(&self, name: &str) -> bool {
        let first = name.chars().next();
        matches!(first, Some(c) if c.is_uppercase())
    }
}

impl Default for SemanticTokensProvider {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Public Token Type/Modifier Lists (for server capabilities)
// ============================================================================

/// Get all semantic token types for LSP capabilities
pub fn semantic_token_types() -> Vec<tower_lsp::lsp_types::SemanticTokenType> {
    vec![
        tower_lsp::lsp_types::SemanticTokenType::NAMESPACE,
        tower_lsp::lsp_types::SemanticTokenType::TYPE,
        tower_lsp::lsp_types::SemanticTokenType::CLASS,
        tower_lsp::lsp_types::SemanticTokenType::ENUM,
        tower_lsp::lsp_types::SemanticTokenType::INTERFACE,
        tower_lsp::lsp_types::SemanticTokenType::STRUCT,
        tower_lsp::lsp_types::SemanticTokenType::TYPE_PARAMETER,
        tower_lsp::lsp_types::SemanticTokenType::PARAMETER,
        tower_lsp::lsp_types::SemanticTokenType::VARIABLE,
        tower_lsp::lsp_types::SemanticTokenType::PROPERTY,
        tower_lsp::lsp_types::SemanticTokenType::ENUM_MEMBER,
        tower_lsp::lsp_types::SemanticTokenType::EVENT,
        tower_lsp::lsp_types::SemanticTokenType::FUNCTION,
        tower_lsp::lsp_types::SemanticTokenType::METHOD,
        tower_lsp::lsp_types::SemanticTokenType::MACRO,
        tower_lsp::lsp_types::SemanticTokenType::KEYWORD,
        tower_lsp::lsp_types::SemanticTokenType::MODIFIER,
        tower_lsp::lsp_types::SemanticTokenType::COMMENT,
        tower_lsp::lsp_types::SemanticTokenType::STRING,
        tower_lsp::lsp_types::SemanticTokenType::NUMBER,
        tower_lsp::lsp_types::SemanticTokenType::REGEXP,
        tower_lsp::lsp_types::SemanticTokenType::OPERATOR,
        tower_lsp::lsp_types::SemanticTokenType::DECORATOR,
        // D-specific types
        tower_lsp::lsp_types::SemanticTokenType::new("effect"),
        tower_lsp::lsp_types::SemanticTokenType::new("unit"),
        tower_lsp::lsp_types::SemanticTokenType::new("refinement"),
        tower_lsp::lsp_types::SemanticTokenType::new("lifetime"),
        tower_lsp::lsp_types::SemanticTokenType::new("label"),
        tower_lsp::lsp_types::SemanticTokenType::new("attribute"),
    ]
}

/// Get all semantic token modifiers for LSP capabilities
pub fn semantic_token_modifiers() -> Vec<tower_lsp::lsp_types::SemanticTokenModifier> {
    vec![
        tower_lsp::lsp_types::SemanticTokenModifier::DECLARATION,
        tower_lsp::lsp_types::SemanticTokenModifier::DEFINITION,
        tower_lsp::lsp_types::SemanticTokenModifier::READONLY,
        tower_lsp::lsp_types::SemanticTokenModifier::STATIC,
        tower_lsp::lsp_types::SemanticTokenModifier::DEPRECATED,
        tower_lsp::lsp_types::SemanticTokenModifier::ABSTRACT,
        tower_lsp::lsp_types::SemanticTokenModifier::ASYNC,
        tower_lsp::lsp_types::SemanticTokenModifier::MODIFICATION,
        tower_lsp::lsp_types::SemanticTokenModifier::DOCUMENTATION,
        tower_lsp::lsp_types::SemanticTokenModifier::DEFAULT_LIBRARY,
        // D-specific modifiers
        tower_lsp::lsp_types::SemanticTokenModifier::new("mutable"),
        tower_lsp::lsp_types::SemanticTokenModifier::new("linear"),
        tower_lsp::lsp_types::SemanticTokenModifier::new("affine"),
        tower_lsp::lsp_types::SemanticTokenModifier::new("pure"),
        tower_lsp::lsp_types::SemanticTokenModifier::new("unsafe"),
        tower_lsp::lsp_types::SemanticTokenModifier::new("pub"),
    ]
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semantic_tokens_builder() {
        let source = "let x = 42;";
        let mut builder = SemanticTokensBuilder::new(source);

        builder.push_range(
            0,
            3,
            SemanticTokenType::Keyword,
            SemanticTokenModifiers::NONE,
        );
        builder.push_range(
            4,
            5,
            SemanticTokenType::Variable,
            SemanticTokenModifiers::DEFINITION,
        );
        builder.push_range(
            8,
            10,
            SemanticTokenType::Number,
            SemanticTokenModifiers::NONE,
        );

        let tokens = builder.build();
        assert_eq!(tokens.data.len(), 3);

        // First token: "let" at (0, 0)
        assert_eq!(tokens.data[0].delta_line, 0);
        assert_eq!(tokens.data[0].delta_start, 0);
        assert_eq!(tokens.data[0].length, 3);

        // Second token: "x" at (0, 4)
        assert_eq!(tokens.data[1].delta_line, 0);
        assert_eq!(tokens.data[1].delta_start, 4);
        assert_eq!(tokens.data[1].length, 1);

        // Third token: "42" at (0, 8)
        assert_eq!(tokens.data[2].delta_line, 0);
        assert_eq!(tokens.data[2].delta_start, 4); // delta from previous
        assert_eq!(tokens.data[2].length, 2);
    }

    #[test]
    fn test_multiline_tokens() {
        let source = "fn foo() {\n    let x = 1;\n}";
        let mut builder = SemanticTokensBuilder::new(source);

        builder.push_range(
            0,
            2,
            SemanticTokenType::Keyword,
            SemanticTokenModifiers::NONE,
        );
        builder.push_range(
            3,
            6,
            SemanticTokenType::Function,
            SemanticTokenModifiers::DEFINITION,
        );
        builder.push_range(
            15,
            18,
            SemanticTokenType::Keyword,
            SemanticTokenModifiers::NONE,
        );

        let tokens = builder.build();
        assert_eq!(tokens.data.len(), 3);

        // "fn" at line 0
        assert_eq!(tokens.data[0].delta_line, 0);

        // "foo" at line 0
        assert_eq!(tokens.data[1].delta_line, 0);

        // "let" at line 1
        assert_eq!(tokens.data[2].delta_line, 1);
    }

    #[test]
    fn test_provider_tokenize() {
        let provider = SemanticTokensProvider::new();
        let source = "fn main() { let x = 42; }";

        let tokens = provider.tokenize(source);
        assert!(!tokens.data.is_empty());
    }

    #[test]
    fn test_known_effects() {
        let provider = SemanticTokensProvider::new();
        assert!(provider.known_effects.contains("IO"));
        assert!(provider.known_effects.contains("Mut"));
        assert!(provider.known_effects.contains("GPU"));
        assert!(!provider.known_effects.contains("Unknown"));
    }

    #[test]
    fn test_known_units() {
        let provider = SemanticTokensProvider::new();
        assert!(provider.known_units.contains("kg"));
        assert!(provider.known_units.contains("mL"));
        assert!(provider.known_units.contains("Hz"));
        assert!(!provider.known_units.contains("xyz"));
    }

    #[test]
    fn test_is_type_name() {
        let provider = SemanticTokensProvider::new();
        assert!(provider.is_type_name("Vec"));
        assert!(provider.is_type_name("MyStruct"));
        assert!(!provider.is_type_name("my_var"));
        assert!(!provider.is_type_name("_unused"));
    }

    #[test]
    fn test_token_modifiers_combine() {
        let mods = SemanticTokenModifiers::DEFINITION | SemanticTokenModifiers::MUTABLE;
        assert_eq!(
            mods.bits(),
            SemanticTokenModifiers::DEFINITION.bits() | SemanticTokenModifiers::MUTABLE.bits()
        );
    }
}
