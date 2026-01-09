# Parser

The Sounio parser transforms a token stream into an Abstract Syntax Tree (AST). It is located in `compiler/src/parser/`.

## Overview

The parser uses a combination of techniques:

- **Recursive descent** for top-level items, statements, and most language constructs
- **Pratt parsing** for expressions with proper operator precedence
- **Error recovery** for resilient parsing that collects multiple errors

## Entry Points

```rust
/// Parse a token stream into an AST
pub fn parse(tokens: &[Token], source: &str) -> Result<Ast>

/// Parse with a custom NodeId start (for incremental parsing)
pub fn parse_with_id_start(tokens: &[Token], source: &str, start_id: u32) -> Result<(Ast, u32)>
```

## Parser State

The parser maintains several pieces of state:

```rust
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    id_gen: IdGenerator,

    /// When false, don't parse `Ident { ... }` as a struct literal
    /// (needed for match expressions)
    allow_struct_literals: bool,

    /// Mapping from NodeId to source spans
    node_spans: HashMap<NodeId, Span>,

    /// Pending `>` from splitting a `>>` token
    /// (for nested generics like `Option<Box<T>>`)
    pending_gt: bool,

    /// Source text for newline detection
    source: &'a str,
}
```

## Parsing Strategy

### Program Structure

```rust
fn parse_program(&mut self) -> Result<Ast> {
    // Optional module declaration: `module foo;`
    let module_name = ...;

    // Parse items until EOF
    while !self.at(TokenKind::Eof) {
        items.push(self.parse_item()?);
    }

    Ok(Ast { module_name, items, node_spans })
}
```

### Item Parsing

Items are top-level declarations:

```rust
pub fn parse_item(&mut self) -> Result<Item> {
    // Skip doc comments
    // Check for macro invocation
    // Parse attributes: #[...]
    // Parse visibility: pub
    // Parse modifiers: linear, affine, async, unsafe

    match self.peek() {
        TokenKind::Fn | TokenKind::Kernel => self.parse_fn(...),
        TokenKind::Struct => self.parse_struct(...),
        TokenKind::Enum => self.parse_enum(...),
        TokenKind::Trait => self.parse_trait(...),
        TokenKind::Impl => self.parse_impl(),
        TokenKind::Type => self.parse_type_alias(...),
        TokenKind::Effect => self.parse_effect(...),
        TokenKind::Handler => self.parse_handler(...),
        TokenKind::Import | TokenKind::Use => self.parse_import(...),
        TokenKind::Extern => self.parse_extern(...),
        TokenKind::Ode => self.parse_ode_def(...),
        TokenKind::Pde => self.parse_pde_def(...),
        TokenKind::Causal => self.parse_causal_model_def(...),
        TokenKind::Module => self.parse_module_decl(...),
        // ...
    }
}
```

### Expression Parsing (Pratt Parsing)

Expressions use Pratt parsing for correct precedence:

```rust
fn parse_expr(&mut self) -> Result<Expr> {
    self.parse_expr_bp(0)  // Start with minimum binding power
}

fn parse_expr_bp(&mut self, min_bp: u8) -> Result<Expr> {
    // Parse prefix (unary operators, atoms)
    let mut lhs = self.parse_prefix()?;

    loop {
        // Get infix operator and its binding power
        let (l_bp, r_bp) = match infix_binding_power(self.peek()) {
            Some((l, r)) => (l, r),
            None => break,
        };

        // Stop if operator has lower precedence
        if l_bp < min_bp {
            break;
        }

        // Parse the operator and right-hand side
        let op = self.advance();
        let rhs = self.parse_expr_bp(r_bp)?;

        lhs = Expr::Binary { op, left: lhs, right: rhs };
    }

    lhs
}
```

### Operator Precedence

Binding powers (higher = tighter binding):

| Level | Operators | Associativity |
|-------|-----------|---------------|
| 1 | `=`, `+=`, `-=`, etc. | Right |
| 2 | `\|\|` | Left |
| 3 | `&&` | Left |
| 4 | `\|` | Left |
| 5 | `^` | Left |
| 6 | `&` | Left |
| 7 | `==`, `!=` | Left |
| 8 | `<`, `<=`, `>`, `>=` | Left |
| 9 | `<<`, `>>` | Left |
| 10 | `+-` (uncertainty) | Left |
| 11 | `++` (concatenation) | Left |
| 12 | `+`, `-` | Left |
| 13 | `*`, `/`, `%` | Left |
| 14 | `as` (cast) | Left |
| 15 | Unary `-`, `!`, `&`, `*` | Right (prefix) |
| 16 | `.`, `()`, `[]` | Left (postfix) |

## Special Parsing Challenges

### Generic Parameter Ambiguity

The `>>` token must be split for nested generics:

```sio
let x: Option<Box<i32>>  // >> must become > >
```

The parser handles this with a `pending_gt` flag:

```rust
fn expect_gt(&mut self) -> Result<()> {
    if self.pending_gt {
        self.pending_gt = false;
        Ok(())
    } else if self.at(TokenKind::Gt) {
        self.advance();
        Ok(())
    } else if self.at(TokenKind::Shr) {
        self.advance();
        self.pending_gt = true;  // Save the second > for later
        Ok(())
    } else {
        Err(...)
    }
}
```

### Struct Literal vs. Block Ambiguity

In match expressions, `{ }` could be a struct literal or a block:

```sio
match x {
    Foo { y } => ...  // Struct pattern, not literal
}
```

The parser disables struct literal parsing in certain contexts:

```rust
self.allow_struct_literals = false;
let pattern = self.parse_pattern()?;
self.allow_struct_literals = true;
```

### Newline-Aware Parsing

Function calls on a new line should not be parsed as calls:

```sio
let x = foo
(1, 2, 3)  // This is a tuple, not a call
```

The parser detects newlines:

```rust
fn had_newline_before_current(&self) -> bool {
    if self.pos == 0 || self.source.is_empty() {
        return false;
    }
    let prev_end = self.tokens.get(self.pos - 1).map(|t| t.span.end);
    let curr_start = self.current().span.start;
    self.source[prev_end..curr_start].contains('\n')
}
```

## Error Recovery

The parser implements panic-mode error recovery to continue parsing after errors.

### Recovery State

```rust
pub struct RecoveryState {
    pub in_recovery: bool,
    pub errors: Vec<ParseError>,
    pub max_errors: usize,
    pub nesting_depth: NestingDepth,
}
```

### Synchronization Points

When an error occurs, the parser advances to a synchronization point:

```rust
// Statement starters
pub const STATEMENT_STARTERS: &[TokenKind] = &[
    TokenKind::Let, TokenKind::Const, TokenKind::If,
    TokenKind::While, TokenKind::For, TokenKind::Loop,
    TokenKind::Return, TokenKind::Break, TokenKind::Continue,
];

// Item starters
pub const ITEM_STARTERS: &[TokenKind] = &[
    TokenKind::Fn, TokenKind::Struct, TokenKind::Enum,
    TokenKind::Type, TokenKind::Impl, TokenKind::Trait,
    TokenKind::Pub, TokenKind::Effect, TokenKind::Handler,
];
```

### Nesting Tracking

The parser tracks bracket nesting to recover properly:

```rust
pub struct NestingDepth {
    pub parens: i32,
    pub braces: i32,
    pub brackets: i32,
}
```

## Parser Error Types

```rust
pub enum ParserError {
    UnexpectedToken {
        span: SourceSpan,
        expected: String,
        found: String,
        context: String,
    },

    RustMutReference {
        span: SourceSpan,
        // Detected &mut (Rust syntax) instead of &! (Sounio syntax)
    },

    InvalidModuleLevelItem {
        span: SourceSpan,
        help: Option<String>,
    },

    // ... more error types
}
```

## Parsing Specific Constructs

### Functions

```rust
fn parse_fn(&mut self, visibility, modifiers, attributes) -> Result<Item> {
    // kernel fn or fn
    let is_kernel = self.at(TokenKind::Kernel);
    if is_kernel { self.advance(); }
    self.expect(TokenKind::Fn)?;

    let name = self.parse_ident()?;
    let generics = self.parse_generics()?;
    self.expect(TokenKind::LParen)?;
    let params = self.parse_params()?;
    self.expect(TokenKind::RParen)?;

    // Return type
    let return_type = if self.at(TokenKind::Arrow) {
        self.advance();
        Some(self.parse_type()?)
    } else {
        None
    };

    // Effects: with IO, Mut
    let effects = if self.at(TokenKind::With) {
        self.advance();
        self.parse_effect_list()?
    } else {
        vec![]
    };

    // Where clause
    let where_clause = self.parse_where_clause()?;

    // Body
    let body = self.parse_block()?;

    Ok(Item::Function(FnDef { ... }))
}
```

### Types

```rust
fn parse_type(&mut self) -> Result<TypeExpr> {
    match self.peek() {
        // Unit type: ()
        TokenKind::LParen if self.peek_n(1) == TokenKind::RParen => {
            self.advance();
            self.advance();
            Ok(TypeExpr::Unit)
        }

        // Reference: &T or &!T
        TokenKind::Amp => {
            self.advance();
            let mutable = self.at(TokenKind::Bang);
            if mutable { self.advance(); }
            let inner = Box::new(self.parse_type()?);
            Ok(TypeExpr::Reference { mutable, inner })
        }

        // Array: [T] or [T; N]
        TokenKind::LBracket => {
            self.advance();
            let element = Box::new(self.parse_type()?);
            let size = if self.at(TokenKind::Semi) {
                self.advance();
                Some(Box::new(self.parse_expr()?))
            } else {
                None
            };
            self.expect(TokenKind::RBracket)?;
            Ok(TypeExpr::Array { element, size })
        }

        // Named type with optional generics: Foo<T, U>
        TokenKind::Ident => {
            let path = self.parse_path()?;
            let args = if self.at(TokenKind::Lt) {
                self.parse_generic_args()?
            } else {
                vec![]
            };
            Ok(TypeExpr::Named { path, args, unit: None })
        }

        // Epistemic types: Knowledge[T, ...]
        TokenKind::Knowledge => self.parse_knowledge_type(),

        // ... more type forms
    }
}
```

### Patterns

```rust
fn parse_pattern(&mut self) -> Result<Pattern> {
    match self.peek() {
        // Wildcard: _
        TokenKind::Underscore => {
            self.advance();
            Ok(Pattern::Wildcard)
        }

        // Literal patterns
        TokenKind::IntLit | TokenKind::FloatLit | TokenKind::StringLit |
        TokenKind::True | TokenKind::False => {
            let lit = self.parse_literal()?;
            Ok(Pattern::Literal(lit))
        }

        // Variable binding or enum pattern
        TokenKind::Ident => {
            let path = self.parse_path()?;
            if self.at(TokenKind::LBrace) {
                // Struct pattern: S { x, y }
                self.parse_struct_pattern(path)
            } else if self.at(TokenKind::LParen) {
                // Enum variant: E::V(x, y)
                self.parse_enum_pattern(path)
            } else {
                // Variable binding
                Ok(Pattern::Binding {
                    name: path.segments.last().unwrap().clone(),
                    mutable: false,
                })
            }
        }

        // Tuple pattern: (a, b, c)
        TokenKind::LParen => self.parse_tuple_pattern(),

        // ... more pattern forms
    }
}
```

## Files

- `compiler/src/parser/mod.rs` - Main parser implementation
- `compiler/src/parser/errors.rs` - Error types and messages
- `compiler/src/parser/recovery.rs` - Error recovery infrastructure
- `compiler/src/parser/tests/` - Parser tests
