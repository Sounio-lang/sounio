//! Token definitions for the Sounio lexer

use crate::common::Span;
use logos::Logos;
use serde::{Deserialize, Serialize};

/// A token with its kind, span, and text
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

/// Token kinds recognized by the lexer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Logos, Serialize, Deserialize)]
#[logos(skip r"[ \t\r\n\f]+")]
// Skip regular comments but NOT doc comments (captured as tokens below)
// Regular line comments: // optionally followed by content (but not / or ! as first char for doc comments)
// Matches: // (empty), // text, //   spaces, but not /// or //!
#[logos(skip r"//([^/!\n][^\n]*)?")]
// Block comments that aren't doc comments: /* not followed by * or !
// Matches: /* text */, /*  */, but not /** or /*!
#[logos(skip r"/\*([^*!]([^*]|\*[^/])*|[^*!]?)\*/")]
pub enum TokenKind {
    // Keywords
    #[token("module")]
    Module,
    #[token("import")]
    Import,
    #[token("use")]
    Use,
    #[token("export")]
    Export,
    #[token("fn")]
    Fn,
    #[token("let")]
    Let,
    #[token("var")]
    Var,
    #[token("mut")]
    Mut,
    #[token("const")]
    Const,
    #[token("type")]
    Type,
    #[token("struct")]
    Struct,
    #[token("enum")]
    Enum,
    #[token("trait")]
    Trait,
    #[token("impl")]
    Impl,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("match")]
    Match,
    #[token("for")]
    For,
    #[token("while")]
    While,
    #[token("loop")]
    Loop,
    #[token("break")]
    Break,
    #[token("continue")]
    Continue,
    #[token("return")]
    Return,
    #[token("in")]
    In,
    #[token("as")]
    As,
    #[token("where")]
    Where,
    #[token("pub")]
    Pub,
    #[token("self")]
    SelfLower,
    #[token("Self")]
    SelfUpper,

    // Ontology keywords
    #[token("ontology")]
    Ontology,
    #[token("from")]
    From,
    #[token("align")]
    Align,
    #[token("distance")]
    Distance,
    #[token("threshold")]
    Threshold,
    #[token("compat")]
    Compat,

    // D-specific keywords
    #[token("effect")]
    Effect,
    #[token("handler")]
    Handler,
    #[token("handle")]
    Handle,
    #[token("with")]
    With,
    #[token("perform")]
    Perform,
    #[token("resume")]
    Resume,
    #[token("linear")]
    Linear,
    #[token("affine")]
    Affine,
    #[token("move")]
    Move,
    #[token("copy")]
    Copy,
    #[token("drop")]
    Drop,
    #[token("kernel")]
    Kernel,
    #[token("tile")]
    Tile,
    #[token("device")]
    Device,
    #[token("shared")]
    Shared,
    #[token("gpu")]
    Gpu,
    #[token("async")]
    Async,
    #[token("await")]
    Await,
    #[token("spawn")]
    Spawn,
    #[token("sample")]
    Sample,
    #[token("observe")]
    Observe,
    #[token("infer")]
    Infer,
    #[token("proof")]
    Proof,

    // Scientific DSL keywords
    #[token("ode")]
    Ode,
    #[token("pde")]
    Pde,
    #[token("causal")]
    Causal,
    #[token("nodes")]
    Nodes,
    #[token("edges")]
    Edges,
    #[token("equations")]
    Equations,
    #[token("state")]
    State,
    #[token("params")]
    Params,
    #[token("domain")]
    Domain,
    #[token("boundary")]
    Boundary,
    #[token("initial")]
    Initial,

    // Epistemic/causal keywords
    #[token("Knowledge")]
    Knowledge,
    #[token("Quantity")]
    Quantity,
    #[token("Tensor")]
    Tensor,

    // Linear algebra primitive types
    #[token("vec2")]
    Vec2,
    #[token("vec3")]
    Vec3,
    #[token("vec4")]
    Vec4,
    #[token("mat2")]
    Mat2,
    #[token("mat3")]
    Mat3,
    #[token("mat4")]
    Mat4,
    #[token("quat")]
    Quat,

    // Automatic differentiation types
    #[token("dual")]
    Dual,
    #[token("grad")]
    Grad,
    #[token("jacobian")]
    Jacobian,
    #[token("hessian")]
    Hessian,
    #[token("OntologyTerm")]
    OntologyTerm,
    #[token("do")]
    Do,
    #[token("counterfactual")]
    Counterfactual,
    #[token("Valid")]
    Valid,
    #[token("ValidUntil")]
    ValidUntil,
    #[token("ValidWhile")]
    ValidWhile,
    #[token("Derived")]
    Derived,
    #[token("Source")]
    SourceProv,
    #[token("Computed")]
    Computed,
    #[token("Literature")]
    Literature,
    #[token("Measured")]
    Measured,
    #[token("Input")]
    InputProv,
    #[token("query")]
    Query,
    #[token("invariant")]
    Invariant,
    #[token("requires")]
    Requires,
    #[token("ensures")]
    Ensures,
    #[token("assert")]
    Assert,
    #[token("assume")]
    Assume,
    #[token("unsafe")]
    Unsafe,
    #[token("extern")]
    Extern,
    #[token("static")]
    Static,

    // Boolean literals
    #[token("true")]
    True,
    #[token("false")]
    False,

    // Literals
    #[regex(r"[0-9][0-9_]*", priority = 2)]
    IntLit,
    #[regex(r"0x[0-9a-fA-F][0-9a-fA-F_]*")]
    HexLit,
    #[regex(r"0b[01][01_]*")]
    BinLit,
    #[regex(r"0o[0-7][0-7_]*")]
    OctLit,
    // Float literals: supports 3.14, 3.14e10, 3.14e-10, 1e10, 1e-10
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?|[0-9][0-9_]*[eE][+-]?[0-9]+")]
    FloatLit,
    #[regex(r#""([^"\\]|\\.)*""#)]
    StringLit,
    #[regex(r#"'([^'\\]|\\.)'"#)]
    CharLit,

    // Unit literals (number with underscore-prefixed unit suffix)
    // e.g., 500_mg, 10.5_mL, 3.14_kg
    #[regex(r"[0-9][0-9_]*_[a-zA-Z][a-zA-Z0-9_/]*", priority = 3)]
    IntUnitLit,
    #[regex(
        r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?_[a-zA-Z][a-zA-Z0-9_/]*",
        priority = 3
    )]
    FloatUnitLit,

    // Identifiers (priority 1 so _ token takes precedence)
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", priority = 1)]
    Ident,

    // Operators
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("^")]
    Caret,
    #[token("&")]
    Amp,
    #[token("|")]
    Pipe,
    #[token("~")]
    Tilde,
    #[token("!")]
    Bang,
    #[token("=")]
    Eq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,

    // Compound operators
    #[token("==")]
    EqEq,
    #[token("!=")]
    Ne,
    #[token("<=")]
    Le,
    #[token(">=")]
    Ge,
    #[token("&&")]
    AmpAmp,
    #[token("||")]
    PipePipe,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,
    #[token("++")]
    PlusPlus,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("&=")]
    AmpEq,
    #[token("|=")]
    PipeEq,
    #[token("^=")]
    CaretEq,
    #[token("<<=")]
    ShlEq,
    #[token(">>=")]
    ShrEq,

    // Arrows and special operators
    #[token("->")]
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("<-")]
    LeftArrow,

    // Uncertainty operator (plus-minus)
    #[token("+-")]
    PlusMinus,

    // Partial derivative (for ODE/PDE DSL)
    #[regex(r"∂|\\partial")]
    Partial,

    // Delimiters
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    // Punctuation
    #[token(",")]
    Comma,
    #[token(";")]
    Semi,
    #[token(":")]
    Colon,
    #[token("::")]
    ColonColon,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,
    #[token("...")]
    DotDotDot,
    #[token("..=")]
    DotDotEq,
    #[token("@")]
    At,
    #[token("#")]
    Hash,
    #[token("$")]
    Dollar,
    #[token("?")]
    Question,
    #[token("_", priority = 2)]
    Underscore,

    // Documentation comments
    /// Outer doc comment: /// ...
    #[regex(r"///[^\n]*")]
    DocCommentOuter,
    /// Inner doc comment: //! ...
    #[regex(r"//![^\n]*")]
    DocCommentInner,
    /// Outer block doc comment: /** ... */
    #[regex(r"/\*\*([^*]|\*[^/])*\*/")]
    DocBlockOuter,
    /// Inner block doc comment: /*! ... */
    #[regex(r"/\*!([^*]|\*[^/])*\*/")]
    DocBlockInner,

    // Special
    Eof,
}

impl TokenKind {
    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            TokenKind::Module
                | TokenKind::Import
                | TokenKind::Use
                | TokenKind::Export
                | TokenKind::Fn
                | TokenKind::Let
                | TokenKind::Var
                | TokenKind::Mut
                | TokenKind::Const
                | TokenKind::Type
                | TokenKind::Struct
                | TokenKind::Enum
                | TokenKind::Trait
                | TokenKind::Impl
                | TokenKind::If
                | TokenKind::Else
                | TokenKind::Match
                | TokenKind::For
                | TokenKind::While
                | TokenKind::Loop
                | TokenKind::Break
                | TokenKind::Continue
                | TokenKind::Return
                | TokenKind::In
                | TokenKind::As
                | TokenKind::Where
                | TokenKind::Pub
                | TokenKind::SelfLower
                | TokenKind::SelfUpper
                | TokenKind::Effect
                | TokenKind::Handler
                | TokenKind::Handle
                | TokenKind::With
                | TokenKind::Perform
                | TokenKind::Resume
                | TokenKind::Linear
                | TokenKind::Affine
                | TokenKind::Move
                | TokenKind::Copy
                | TokenKind::Drop
                | TokenKind::Kernel
                | TokenKind::Device
                | TokenKind::Shared
                | TokenKind::Gpu
                | TokenKind::Async
                | TokenKind::Await
                | TokenKind::Spawn
                | TokenKind::Sample
                | TokenKind::Observe
                | TokenKind::Infer
                | TokenKind::Proof
                | TokenKind::Invariant
                | TokenKind::Requires
                | TokenKind::Ensures
                | TokenKind::Assert
                | TokenKind::Assume
                | TokenKind::Unsafe
                | TokenKind::Extern
                | TokenKind::True
                | TokenKind::False
                // Epistemic/causal keywords
                | TokenKind::Knowledge
                | TokenKind::Quantity
                | TokenKind::Tensor
                | TokenKind::Vec2
                | TokenKind::Vec3
                | TokenKind::Vec4
                | TokenKind::Mat2
                | TokenKind::Mat3
                | TokenKind::Mat4
                | TokenKind::Quat
                | TokenKind::Dual
                | TokenKind::Grad
                | TokenKind::Jacobian
                | TokenKind::Hessian
                | TokenKind::OntologyTerm
                | TokenKind::Do
                | TokenKind::Counterfactual
                | TokenKind::Valid
                | TokenKind::ValidUntil
                | TokenKind::ValidWhile
                | TokenKind::Derived
                | TokenKind::SourceProv
                | TokenKind::Computed
                | TokenKind::Literature
                | TokenKind::Measured
                | TokenKind::InputProv
                | TokenKind::Query
                // Ontology keywords
                | TokenKind::Ontology
                | TokenKind::From
                | TokenKind::Align
                | TokenKind::Distance
                | TokenKind::Threshold
                | TokenKind::Compat
                // Scientific DSL keywords
                | TokenKind::Ode
                | TokenKind::Pde
                | TokenKind::Causal
                | TokenKind::Nodes
                | TokenKind::Edges
                | TokenKind::Equations
        )
    }

    /// Check if this token is a literal
    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            TokenKind::IntLit
                | TokenKind::HexLit
                | TokenKind::BinLit
                | TokenKind::OctLit
                | TokenKind::FloatLit
                | TokenKind::StringLit
                | TokenKind::CharLit
                | TokenKind::IntUnitLit
                | TokenKind::FloatUnitLit
                | TokenKind::True
                | TokenKind::False
        )
    }

    /// Check if this token is an operator
    pub fn is_operator(&self) -> bool {
        matches!(
            self,
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
        )
    }

    /// Get the string representation of the token
    pub fn as_str(&self) -> &'static str {
        match self {
            TokenKind::Module => "module",
            TokenKind::Import => "import",
            TokenKind::Use => "use",
            TokenKind::Export => "export",
            TokenKind::Fn => "fn",
            TokenKind::Let => "let",
            TokenKind::Var => "var",
            TokenKind::Mut => "mut",
            TokenKind::Const => "const",
            TokenKind::Type => "type",
            TokenKind::Struct => "struct",
            TokenKind::Enum => "enum",
            TokenKind::Trait => "trait",
            TokenKind::Impl => "impl",
            TokenKind::If => "if",
            TokenKind::Else => "else",
            TokenKind::Match => "match",
            TokenKind::For => "for",
            TokenKind::While => "while",
            TokenKind::Loop => "loop",
            TokenKind::Break => "break",
            TokenKind::Continue => "continue",
            TokenKind::Return => "return",
            TokenKind::In => "in",
            TokenKind::As => "as",
            TokenKind::Where => "where",
            TokenKind::Pub => "pub",
            TokenKind::SelfLower => "self",
            TokenKind::SelfUpper => "Self",
            TokenKind::Effect => "effect",
            TokenKind::Handler => "handler",
            TokenKind::Handle => "handle",
            TokenKind::With => "with",
            TokenKind::Perform => "perform",
            TokenKind::Resume => "resume",
            TokenKind::Linear => "linear",
            TokenKind::Affine => "affine",
            TokenKind::Move => "move",
            TokenKind::Copy => "copy",
            TokenKind::Drop => "drop",
            TokenKind::Kernel => "kernel",
            TokenKind::Tile => "tile",
            TokenKind::Device => "device",
            TokenKind::Shared => "shared",
            TokenKind::Gpu => "gpu",
            TokenKind::Async => "async",
            TokenKind::Await => "await",
            TokenKind::Spawn => "spawn",
            TokenKind::Sample => "sample",
            TokenKind::Observe => "observe",
            TokenKind::Infer => "infer",
            TokenKind::Proof => "proof",
            TokenKind::Invariant => "invariant",
            TokenKind::Requires => "requires",
            TokenKind::Ensures => "ensures",
            TokenKind::Assert => "assert",
            TokenKind::Assume => "assume",
            TokenKind::Unsafe => "unsafe",
            TokenKind::Extern => "extern",
            TokenKind::Static => "static",
            TokenKind::True => "true",
            TokenKind::False => "false",
            TokenKind::IntLit => "<int>",
            TokenKind::HexLit => "<hex>",
            TokenKind::BinLit => "<bin>",
            TokenKind::OctLit => "<oct>",
            TokenKind::FloatLit => "<float>",
            TokenKind::StringLit => "<string>",
            TokenKind::CharLit => "<char>",
            TokenKind::IntUnitLit => "<int_unit>",
            TokenKind::FloatUnitLit => "<float_unit>",
            TokenKind::Ident => "<ident>",
            TokenKind::Plus => "+",
            TokenKind::Minus => "-",
            TokenKind::Star => "*",
            TokenKind::Slash => "/",
            TokenKind::Percent => "%",
            TokenKind::Caret => "^",
            TokenKind::Amp => "&",
            TokenKind::Pipe => "|",
            TokenKind::Tilde => "~",
            TokenKind::Bang => "!",
            TokenKind::Eq => "=",
            TokenKind::Lt => "<",
            TokenKind::Gt => ">",
            TokenKind::EqEq => "==",
            TokenKind::Ne => "!=",
            TokenKind::Le => "<=",
            TokenKind::Ge => ">=",
            TokenKind::AmpAmp => "&&",
            TokenKind::PipePipe => "||",
            TokenKind::Shl => "<<",
            TokenKind::Shr => ">>",
            TokenKind::PlusPlus => "++",
            TokenKind::PlusEq => "+=",
            TokenKind::MinusEq => "-=",
            TokenKind::StarEq => "*=",
            TokenKind::SlashEq => "/=",
            TokenKind::PercentEq => "%=",
            TokenKind::AmpEq => "&=",
            TokenKind::PipeEq => "|=",
            TokenKind::CaretEq => "^=",
            TokenKind::ShlEq => "<<=",
            TokenKind::ShrEq => ">>=",
            TokenKind::Arrow => "->",
            TokenKind::FatArrow => "=>",
            TokenKind::LeftArrow => "<-",
            TokenKind::LParen => "(",
            TokenKind::RParen => ")",
            TokenKind::LBracket => "[",
            TokenKind::RBracket => "]",
            TokenKind::LBrace => "{",
            TokenKind::RBrace => "}",
            TokenKind::Comma => ",",
            TokenKind::Semi => ";",
            TokenKind::Colon => ":",
            TokenKind::ColonColon => "::",
            TokenKind::Dot => ".",
            TokenKind::DotDot => "..",
            TokenKind::DotDotDot => "...",
            TokenKind::DotDotEq => "..=",
            TokenKind::At => "@",
            TokenKind::Hash => "#",
            TokenKind::Dollar => "$",
            TokenKind::Question => "?",
            TokenKind::Underscore => "_",
            TokenKind::DocCommentOuter => "<doc_comment>",
            TokenKind::DocCommentInner => "<doc_comment_inner>",
            TokenKind::DocBlockOuter => "<doc_block>",
            TokenKind::DocBlockInner => "<doc_block_inner>",
            TokenKind::Eof => "<eof>",
            // Epistemic/causal keywords
            TokenKind::Knowledge => "Knowledge",
            TokenKind::Quantity => "Quantity",
            TokenKind::Tensor => "Tensor",
            TokenKind::Vec2 => "vec2",
            TokenKind::Vec3 => "vec3",
            TokenKind::Vec4 => "vec4",
            TokenKind::Mat2 => "mat2",
            TokenKind::Mat3 => "mat3",
            TokenKind::Mat4 => "mat4",
            TokenKind::Quat => "quat",
            TokenKind::Dual => "dual",
            TokenKind::Grad => "grad",
            TokenKind::Jacobian => "jacobian",
            TokenKind::Hessian => "hessian",
            TokenKind::OntologyTerm => "OntologyTerm",
            TokenKind::Do => "do",
            TokenKind::Counterfactual => "counterfactual",
            TokenKind::Valid => "Valid",
            TokenKind::ValidUntil => "ValidUntil",
            TokenKind::ValidWhile => "ValidWhile",
            TokenKind::Derived => "Derived",
            TokenKind::SourceProv => "Source",
            TokenKind::Computed => "Computed",
            TokenKind::Literature => "Literature",
            TokenKind::Measured => "Measured",
            TokenKind::InputProv => "Input",
            TokenKind::Query => "query",
            // Ontology keywords
            TokenKind::Ontology => "ontology",
            TokenKind::From => "from",
            TokenKind::Align => "align",
            TokenKind::Distance => "distance",
            TokenKind::Threshold => "threshold",
            TokenKind::Compat => "compat",
            TokenKind::Ode => "ode",
            TokenKind::Pde => "pde",
            TokenKind::Causal => "causal",
            TokenKind::Nodes => "nodes",
            TokenKind::Edges => "edges",
            TokenKind::Equations => "equations",
            TokenKind::State => "state",
            TokenKind::Params => "params",
            TokenKind::Domain => "domain",
            TokenKind::Boundary => "boundary",
            TokenKind::Initial => "initial",
            TokenKind::PlusMinus => "±",
            TokenKind::Partial => "∂",
        }
    }
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
