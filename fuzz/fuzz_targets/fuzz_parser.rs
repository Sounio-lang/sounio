//! Fuzz target for the Sounio parser
//!
//! Tests that the parser handles arbitrary token streams without panicking.
//! Uses structured fuzzing to generate syntactically plausible programs.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sounio::{lexer, parser};

/// Structured input for generating Sounio-like source code
#[derive(Debug, Arbitrary)]
struct FuzzInput {
    /// Source code fragments to combine
    fragments: Vec<Fragment>,
}

/// A code fragment that can be combined with others
#[derive(Debug, Arbitrary)]
enum Fragment {
    /// A keyword
    Keyword(Keyword),
    /// An identifier-like string
    Identifier(SmallIdent),
    /// A literal value
    Literal(Literal),
    /// An operator
    Operator(Operator),
    /// A delimiter
    Delimiter(Delimiter),
    /// Whitespace
    Whitespace(Whitespace),
    /// A complete statement
    Statement(Statement),
}

#[derive(Debug, Arbitrary)]
enum Keyword {
    Fn,
    Let,
    Var,
    If,
    Else,
    Match,
    For,
    While,
    Return,
    Struct,
    Enum,
    Trait,
    Impl,
    Type,
    Pub,
    Module,
    Import,
    With,
    Linear,
    Affine,
    Effect,
    Handler,
    Kernel,
    Async,
    Await,
}

#[derive(Debug, Arbitrary)]
struct SmallIdent {
    /// First char must be letter or underscore
    first: u8,
    /// Rest can include digits
    rest: [u8; 4],
    /// How many of rest to use (0-4)
    len: u8,
}

#[derive(Debug, Arbitrary)]
enum Literal {
    Int(u32),
    Float { mantissa: u16, exp: i8 },
    String(SmallString),
    Bool(bool),
}

#[derive(Debug, Arbitrary)]
struct SmallString {
    chars: [u8; 8],
    len: u8,
}

#[derive(Debug, Arbitrary)]
enum Operator {
    Plus,
    Minus,
    Star,
    Slash,
    Eq,
    EqEq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Arrow,
    FatArrow,
    Colon,
    ColonColon,
    Amp,
    AmpBang, // &! for mutable references in Sounio
    Pipe,
    Dot,
}

#[derive(Debug, Arbitrary)]
enum Delimiter {
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
    Comma,
    Semi,
}

#[derive(Debug, Arbitrary)]
enum Whitespace {
    Space,
    Newline,
    Tab,
}

#[derive(Debug, Arbitrary)]
enum Statement {
    /// let x = expr
    LetBinding { name: SmallIdent, has_type: bool },
    /// fn name(args) -> ret { body }
    FnDef { name: SmallIdent, arg_count: u8 },
    /// struct Name { fields }
    StructDef { name: SmallIdent, field_count: u8 },
    /// if cond { } else { }
    IfElse { has_else: bool },
    /// match x { }
    Match { arm_count: u8 },
}

impl FuzzInput {
    fn to_source(&self) -> String {
        let mut source = String::new();
        for fragment in &self.fragments {
            source.push_str(&fragment.to_string());
        }
        source
    }
}

impl Fragment {
    fn to_string(&self) -> String {
        match self {
            Fragment::Keyword(kw) => kw.to_string(),
            Fragment::Identifier(id) => id.to_string(),
            Fragment::Literal(lit) => lit.to_string(),
            Fragment::Operator(op) => op.to_string(),
            Fragment::Delimiter(d) => d.to_string(),
            Fragment::Whitespace(ws) => ws.to_string(),
            Fragment::Statement(stmt) => stmt.to_string(),
        }
    }
}

impl Keyword {
    fn to_string(&self) -> String {
        match self {
            Keyword::Fn => "fn".to_string(),
            Keyword::Let => "let".to_string(),
            Keyword::Var => "var".to_string(),
            Keyword::If => "if".to_string(),
            Keyword::Else => "else".to_string(),
            Keyword::Match => "match".to_string(),
            Keyword::For => "for".to_string(),
            Keyword::While => "while".to_string(),
            Keyword::Return => "return".to_string(),
            Keyword::Struct => "struct".to_string(),
            Keyword::Enum => "enum".to_string(),
            Keyword::Trait => "trait".to_string(),
            Keyword::Impl => "impl".to_string(),
            Keyword::Type => "type".to_string(),
            Keyword::Pub => "pub".to_string(),
            Keyword::Module => "module".to_string(),
            Keyword::Import => "import".to_string(),
            Keyword::With => "with".to_string(),
            Keyword::Linear => "linear".to_string(),
            Keyword::Affine => "affine".to_string(),
            Keyword::Effect => "effect".to_string(),
            Keyword::Handler => "handler".to_string(),
            Keyword::Kernel => "kernel".to_string(),
            Keyword::Async => "async".to_string(),
            Keyword::Await => "await".to_string(),
        }
    }
}

impl SmallIdent {
    fn to_string(&self) -> String {
        let mut s = String::new();
        // First char: letter or underscore
        let first = match self.first % 53 {
            0..=25 => (b'a' + self.first % 26) as char,
            26..=51 => (b'A' + (self.first - 26) % 26) as char,
            _ => '_',
        };
        s.push(first);

        // Rest: alphanumeric or underscore
        let len = (self.len % 5) as usize;
        for i in 0..len {
            let c = self.rest[i];
            let ch = match c % 63 {
                0..=25 => (b'a' + c % 26) as char,
                26..=51 => (b'A' + (c - 26) % 26) as char,
                52..=61 => (b'0' + (c - 52) % 10) as char,
                _ => '_',
            };
            s.push(ch);
        }
        s
    }
}

impl Literal {
    fn to_string(&self) -> String {
        match self {
            Literal::Int(n) => n.to_string(),
            Literal::Float { mantissa, exp } => {
                format!("{}.{}e{}", mantissa / 100, mantissa % 100, exp)
            }
            Literal::String(s) => format!("\"{}\"", s.to_string()),
            Literal::Bool(b) => if *b { "true" } else { "false" }.to_string(),
        }
    }
}

impl SmallString {
    fn to_string(&self) -> String {
        let len = (self.len % 9) as usize;
        self.chars[..len]
            .iter()
            .filter_map(|&c| {
                // Only allow safe printable ASCII characters
                if c >= 32 && c < 127 && c != b'"' && c != b'\\' {
                    Some(c as char)
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Operator {
    fn to_string(&self) -> String {
        match self {
            Operator::Plus => "+".to_string(),
            Operator::Minus => "-".to_string(),
            Operator::Star => "*".to_string(),
            Operator::Slash => "/".to_string(),
            Operator::Eq => "=".to_string(),
            Operator::EqEq => "==".to_string(),
            Operator::Ne => "!=".to_string(),
            Operator::Lt => "<".to_string(),
            Operator::Le => "<=".to_string(),
            Operator::Gt => ">".to_string(),
            Operator::Ge => ">=".to_string(),
            Operator::Arrow => "->".to_string(),
            Operator::FatArrow => "=>".to_string(),
            Operator::Colon => ":".to_string(),
            Operator::ColonColon => "::".to_string(),
            Operator::Amp => "&".to_string(),
            Operator::AmpBang => "&!".to_string(),
            Operator::Pipe => "|".to_string(),
            Operator::Dot => ".".to_string(),
        }
    }
}

impl Delimiter {
    fn to_string(&self) -> String {
        match self {
            Delimiter::LParen => "(".to_string(),
            Delimiter::RParen => ")".to_string(),
            Delimiter::LBrace => "{".to_string(),
            Delimiter::RBrace => "}".to_string(),
            Delimiter::LBracket => "[".to_string(),
            Delimiter::RBracket => "]".to_string(),
            Delimiter::Comma => ",".to_string(),
            Delimiter::Semi => ";".to_string(),
        }
    }
}

impl Whitespace {
    fn to_string(&self) -> String {
        match self {
            Whitespace::Space => " ".to_string(),
            Whitespace::Newline => "\n".to_string(),
            Whitespace::Tab => "\t".to_string(),
        }
    }
}

impl Statement {
    fn to_string(&self) -> String {
        match self {
            Statement::LetBinding { name, has_type } => {
                if *has_type {
                    format!("let {}: i32 = 0;", name.to_string())
                } else {
                    format!("let {} = 0;", name.to_string())
                }
            }
            Statement::FnDef { name, arg_count } => {
                let args: Vec<String> = (0..(*arg_count % 4))
                    .map(|i| format!("arg{}: i32", i))
                    .collect();
                format!("fn {}({}) {{ }}", name.to_string(), args.join(", "))
            }
            Statement::StructDef { name, field_count } => {
                let fields: Vec<String> = (0..(*field_count % 4))
                    .map(|i| format!("    field{}: i32,", i))
                    .collect();
                format!("struct {} {{\n{}\n}}", name.to_string(), fields.join("\n"))
            }
            Statement::IfElse { has_else } => {
                if *has_else {
                    "if true { } else { }".to_string()
                } else {
                    "if true { }".to_string()
                }
            }
            Statement::Match { arm_count } => {
                let arms: Vec<String> = (0..(*arm_count % 4))
                    .map(|i| format!("        {} => {{}}", i))
                    .collect();
                format!("match 0 {{\n{}\n    }}", arms.join(",\n"))
            }
        }
    }
}

fuzz_target!(|input: FuzzInput| {
    let source = input.to_source();

    // First, lex the source
    if let Ok(tokens) = lexer::lex(&source) {
        // Then try to parse - should not panic
        let _ = parser::parse(&tokens, &source);
    }
});
