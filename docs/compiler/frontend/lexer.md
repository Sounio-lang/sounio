# Lexer

The Sounio lexer transforms source code into a stream of tokens. It is located in `compiler/src/lexer/`.

## Overview

The lexer uses the [Logos](https://github.com/maciejhirsz/logos) library for high-performance tokenization. Logos generates an efficient state machine from declarative token definitions.

## Entry Point

```rust
pub fn lex(source: &str) -> Result<Vec<Token>>
```

The `lex` function takes source code as input and returns a vector of tokens, or an error if invalid characters are encountered.

## Token Structure

Each token contains three pieces of information:

```rust
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}
```

- `kind`: The type of token (keyword, operator, literal, etc.)
- `span`: The byte offset range in the source (`start..end`)
- `text`: The original text of the token

## Token Types

The `TokenKind` enum defines all token types recognized by the lexer. Tokens are organized into categories:

### Keywords

Core language keywords:

```rust
#[token("module")] Module,
#[token("import")] Import,
#[token("use")] Use,
#[token("export")] Export,
#[token("fn")] Fn,
#[token("let")] Let,
#[token("var")] Var,
#[token("mut")] Mut,
#[token("const")] Const,
#[token("type")] Type,
#[token("struct")] Struct,
#[token("enum")] Enum,
#[token("trait")] Trait,
#[token("impl")] Impl,
#[token("if")] If,
#[token("else")] Else,
#[token("match")] Match,
#[token("for")] For,
#[token("while")] While,
#[token("loop")] Loop,
#[token("break")] Break,
#[token("continue")] Continue,
#[token("return")] Return,
// ... and more
```

### Effect System Keywords

```rust
#[token("effect")] Effect,
#[token("handler")] Handler,
#[token("handle")] Handle,
#[token("with")] With,
#[token("perform")] Perform,
#[token("resume")] Resume,
```

### Linear/Affine Type Keywords

```rust
#[token("linear")] Linear,
#[token("affine")] Affine,
#[token("move")] Move,
#[token("copy")] Copy,
#[token("drop")] Drop,
```

### GPU Keywords

```rust
#[token("kernel")] Kernel,
#[token("tile")] Tile,
#[token("device")] Device,
#[token("shared")] Shared,
#[token("gpu")] Gpu,
```

### Async Keywords

```rust
#[token("async")] Async,
#[token("await")] Await,
#[token("spawn")] Spawn,
```

### Epistemic/Causal Keywords

```rust
#[token("Knowledge")] Knowledge,
#[token("Quantity")] Quantity,
#[token("Tensor")] Tensor,
#[token("do")] Do,
#[token("counterfactual")] Counterfactual,
#[token("Valid")] Valid,
#[token("ValidUntil")] ValidUntil,
#[token("ValidWhile")] ValidWhile,
#[token("Derived")] Derived,
#[token("Source")] SourceProv,
#[token("Computed")] Computed,
#[token("Literature")] Literature,
#[token("Measured")] Measured,
#[token("Input")] InputProv,
```

### Scientific DSL Keywords

```rust
#[token("ode")] Ode,
#[token("pde")] Pde,
#[token("causal")] Causal,
#[token("nodes")] Nodes,
#[token("edges")] Edges,
#[token("equations")] Equations,
#[token("state")] State,
#[token("params")] Params,
#[token("domain")] Domain,
#[token("boundary")] Boundary,
#[token("initial")] Initial,
```

### Linear Algebra Types

```rust
#[token("vec2")] Vec2,
#[token("vec3")] Vec3,
#[token("vec4")] Vec4,
#[token("mat2")] Mat2,
#[token("mat3")] Mat3,
#[token("mat4")] Mat4,
#[token("quat")] Quat,
```

### Literals

```rust
// Integer literals
#[regex(r"[0-9][0-9_]*", priority = 2)]
IntLit,
#[regex(r"0x[0-9a-fA-F][0-9a-fA-F_]*")]
HexLit,
#[regex(r"0b[01][01_]*")]
BinLit,
#[regex(r"0o[0-7][0-7_]*")]
OctLit,

// Float literals (supports scientific notation)
#[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?|[0-9][0-9_]*[eE][+-]?[0-9]+")]
FloatLit,

// String and character literals
#[regex(r#""([^"\\]|\\.)*""#)]
StringLit,
#[regex(r#"'([^'\\]|\\.)'"#)]
CharLit,

// Boolean literals
#[token("true")] True,
#[token("false")] False,
```

### Unit Literals (Sounio-specific)

Unit literals combine a number with a unit suffix:

```rust
// Integer with unit: 500_mg, 100_km
#[regex(r"[0-9][0-9_]*_[a-zA-Z][a-zA-Z0-9_/]*", priority = 3)]
IntUnitLit,

// Float with unit: 10.5_mL, 3.14_rad
#[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?_[a-zA-Z][a-zA-Z0-9_/]*", priority = 3)]
FloatUnitLit,
```

Examples:
- `500_mg` - 500 milligrams
- `10.5_mL` - 10.5 milliliters
- `5.0_mg/mL` - 5 milligrams per milliliter

### Operators

**Arithmetic:**
```rust
#[token("+")] Plus,
#[token("-")] Minus,
#[token("*")] Star,
#[token("/")] Slash,
#[token("%")] Percent,
#[token("^")] Caret,
```

**Comparison:**
```rust
#[token("==")] EqEq,
#[token("!=")] Ne,
#[token("<")] Lt,
#[token("<=")] Le,
#[token(">")] Gt,
#[token(">=")] Ge,
```

**Logical:**
```rust
#[token("&&")] AmpAmp,
#[token("||")] PipePipe,
#[token("!")] Bang,
```

**Arrows:**
```rust
#[token("->")] Arrow,     // Function return type
#[token("=>")] FatArrow,  // Match arms
#[token("<-")] LeftArrow, // Channel receive
```

**Special:**
```rust
#[token("+-")] PlusMinus,           // Uncertainty: x +- 0.1
#[token("++")] PlusPlus,            // Array concatenation
#[regex(r"[+-]|\\partial")] Partial, // Partial derivative
```

### Delimiters and Punctuation

```rust
#[token("(")] LParen,
#[token(")")] RParen,
#[token("[")] LBracket,
#[token("]")] RBracket,
#[token("{")] LBrace,
#[token("}")] RBrace,
#[token(",")] Comma,
#[token(";")] Semi,
#[token(":")] Colon,
#[token("::")] ColonColon,
#[token(".")] Dot,
#[token("..")] DotDot,
#[token("...")] DotDotDot,
#[token("..=")] DotDotEq,
#[token("@")] At,
#[token("#")] Hash,
#[token("?")] Question,
#[token("_")] Underscore,
```

### Documentation Comments

Unlike regular comments which are skipped, doc comments are captured as tokens:

```rust
// Outer doc comments: /// ...
#[regex(r"///[^\n]*")]
DocCommentOuter,

// Inner doc comments: //! ...
#[regex(r"//![^\n]*")]
DocCommentInner,

// Outer block doc comments: /** ... */
#[regex(r"/\*\*([^*]|\*[^/])*\*/")]
DocBlockOuter,

// Inner block doc comments: /*! ... */
#[regex(r"/\*!([^*]|\*[^/])*\*/")]
DocBlockInner,
```

## Whitespace and Comments

The lexer is configured to skip whitespace and regular comments:

```rust
#[logos(skip r"[ \t\r\n\f]+")]
// Skip regular line comments (not doc comments)
#[logos(skip r"//([^/!\n][^\n]*)?")]
// Skip block comments that aren't doc comments
#[logos(skip r"/\*([^*!]([^*]|\*[^/])*|[^*!]?)\*/")]
```

## Example

Given this source code:

```sio
let dose: mg = 500.0
fn calculate(x: f64) -> f64 with IO { x * 2.0 }
```

The lexer produces:

```
Token { kind: Let,      span: 0..3,   text: "let" }
Token { kind: Ident,    span: 4..8,   text: "dose" }
Token { kind: Colon,    span: 8..9,   text: ":" }
Token { kind: Ident,    span: 10..12, text: "mg" }
Token { kind: Eq,       span: 13..14, text: "=" }
Token { kind: FloatLit, span: 15..20, text: "500.0" }
Token { kind: Fn,       span: 21..23, text: "fn" }
Token { kind: Ident,    span: 24..33, text: "calculate" }
Token { kind: LParen,   span: 33..34, text: "(" }
Token { kind: Ident,    span: 34..35, text: "x" }
Token { kind: Colon,    span: 35..36, text: ":" }
Token { kind: Ident,    span: 37..40, text: "f64" }
Token { kind: RParen,   span: 40..41, text: ")" }
Token { kind: Arrow,    span: 42..44, text: "->" }
Token { kind: Ident,    span: 45..48, text: "f64" }
Token { kind: With,     span: 49..53, text: "with" }
Token { kind: Ident,    span: 54..56, text: "IO" }
Token { kind: LBrace,   span: 57..58, text: "{" }
Token { kind: Ident,    span: 59..60, text: "x" }
Token { kind: Star,     span: 61..62, text: "*" }
Token { kind: FloatLit, span: 63..66, text: "2.0" }
Token { kind: RBrace,   span: 67..68, text: "}" }
Token { kind: Eof,      span: 68..68, text: "" }
```

## Token Utilities

The `TokenKind` enum provides several utility methods:

```rust
impl TokenKind {
    /// Check if this token is a keyword
    pub fn is_keyword(&self) -> bool

    /// Check if this token is a literal
    pub fn is_literal(&self) -> bool

    /// Check if this token is an operator
    pub fn is_operator(&self) -> bool

    /// Get the string representation of the token
    pub fn as_str(&self) -> &'static str
}
```

## Error Handling

If the lexer encounters an unrecognized character, it returns an error:

```rust
Err(miette::miette!(
    "Unexpected character at position {}: {:?}",
    span.start,
    &source[span.clone()]
))
```

## Implementation Details

### Priority Handling

Some token patterns overlap. Logos handles this using priority:

```rust
// Unit literals have higher priority than plain literals
#[regex(r"[0-9][0-9_]*_[a-zA-Z][a-zA-Z0-9_/]*", priority = 3)]
IntUnitLit,

#[regex(r"[0-9][0-9_]*", priority = 2)]
IntLit,
```

### EOF Token

The lexer always appends an EOF token at the end:

```rust
tokens.push(Token {
    kind: TokenKind::Eof,
    span: Span::new(source.len(), source.len()),
    text: String::new(),
});
```

This simplifies parser lookahead by ensuring there's always a token to examine.

## Files

- `compiler/src/lexer/mod.rs` - Main lexer module, `lex()` function
- `compiler/src/lexer/tokens.rs` - `Token` and `TokenKind` definitions
