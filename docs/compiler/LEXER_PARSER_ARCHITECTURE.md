# Lexer e Parser - Arquitetura do Compilador Sounio

## Visão Geral

O frontend do compilador Sounio é dividido em duas camadas principais:

```
Source Code (.sio)
        ↓
┌─────────────────┐
│     Lexer       │  → Tokens com Span e Text
│  (Logos-based)  │
└─────────────────┘
        ↓
┌─────────────────┐
│     Parser      │  → Abstract Syntax Tree (AST)
│ (Recursive Descent)
└─────────────────┘
        ↓
   AST Validada
```

## 1. Lexer

### Localização
- **Arquivo principal**: [`crates/souc/src/lexer/mod.rs`](../../crates/souc/src/lexer/mod.rs)
- **Definição de tokens**: [`crates/souc/src/lexer/tokens.rs`](../../crates/souc/src/lexer/tokens.rs) (799 linhas)

### Arquitetura

O lexer utiliza a biblioteca **Logos** - um gerador de lexer extremamente rápido baseado em regex.

#### Estrutura Principal

```rust
// crates/souc/src/lexer/mod.rs:14
pub fn lex(source: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut lexer = TokenKind::lexer(source);

    while let Some(result) = lexer.next() {
        let span = lexer.span();
        let kind = match result {
            Ok(kind) => kind,
            Err(_) => {
                return Err(miette::miette!(
                    "Unexpected character at position {}: {:?}",
                    span.start,
                    &source[span.clone()]
                ));
            }
        };

        tokens.push(Token {
            kind,
            span: Span::new(span.start, span.end),
            text: source[span].to_string(),
        });
    }

    // Adiciona token EOF
    tokens.push(Token {
        kind: TokenKind::Eof,
        span: Span::new(source.len(), source.len()),
        text: String::new(),
    });

    Ok(tokens)
}
```

### Token Definition

```rust
// crates/souc/src/lexer/tokens.rs:8
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}

// crates/souc/src/lexer/tokens.rs:16
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Logos, Serialize, Deserialize)]
#[logos(skip r"[ \t\r\n\f]+")]  // Skip whitespace
#[logos(skip r"//([^/!\n][^\n]*)?")]  // Skip comments (not doc)
pub enum TokenKind {
    // Keywords (150+ tipos de tokens)
    #[token("module")] Module,
    #[token("fn")] Fn,
    #[token("let")] Let,
    // ... etc
}
```

### Categorias de Tokens

#### Palavras-chave (Keywords)
- **Controle**: `fn`, `let`, `mut`, `const`, `if`, `else`, `match`, `for`, `while`, `loop`
- **Tipos**: `struct`, `enum`, `trait`, `impl`, `type`
- **Efeitos**: `effect`, `handler`, `handle`, `with`, `perform`, `resume`
- **Concorrência**: `async`, `await`, `spawn`
- **CIência**: `ode`, `pde`, `causal`, `Knowledge`, `quat`, `dual`
- **Ontologia**: `ontology`, `align`, `compat`, `threshold`

#### Literais
```rust
// Inteiros: 42, 0xFF, 0b1010, 0o755
#[regex(r"[0-9][0-9_]*", priority = 2)]
IntLit,

// Floats: 3.14, 1e10, 3.14e-10
#[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9]+)?")]
FloatLit,

// Strings: "hello"
#[regex(r#""([^"\\]|\\.)*""#)]
StringLit,

// C strings: c"hello" (null-terminated para FFI)
#[regex(r#"c"([^"\\]|\\.)*""#, priority = 2)]
CStringLit,

// Unit literals: 500_mg, 10.5_mL
#[regex(r"[0-9][0-9_]*_[a-zA-Z][a-zA-Z0-9_/]*", priority = 3)]
IntUnitLit,
```

#### Operadores
```rust
// Aritméticos: +, -, *, /, %
#[token("+")] Plus,
#[token("-")] Minus,
#[token("*")] Star,
#[token("/")] Slash,

// Comparação: ==, !=, <=, >=
#[token("==")] EqEq,
#[token("!=")] Ne,
#[token("<=")] Le,
#[token(">=")] Ge,

// Lógicos: &&, ||
#[token("&&")] AmpAmp,
#[token("||")] PipePipe,

// Bits: <<, >>
#[token("<<")] Shl,
#[token(">>")] Shr,
```

#### Doc Comments
```rust
// Outer doc: /// This is a doc comment
#[token("///...")] DocCommentOuter,

// Inner doc: //! Module-level doc
#[token("//!...")] DocCommentInner,

// Block docs: /** ... */
#[token("/**...*/")] DocBlockOuter,
#[token("/*!...*/")] DocBlockInner,
```

### Features Especiais

#### Literais com Unidades
```sio
// O lexer tokeniza como:
// 500_mg → TokenKind::IntUnitLit (texto: "500_mg")
// 10.5_mL → TokenKind::FloatUnitLit (texto: "10.5_mL")
// 5.0_mg/mL → TokenKind::FloatUnitLit (texto: "5.0_mg/mL")

let dose = 500_mg + 200_mg;
let conc = 10.5_mL / 5.0_mg;
```

#### Splitting de `>>` para Generics
```sio
// Option<Box<T>> → parsed como Option < Box < T > >
// O lexer gera TokenKind::Shr (>>) que o parser divide
```

### Error Handling

O lexer reporta erros usando `miette` com:
- Posição do caractere inválido
- Contexto do erro
- Sugestões para correções comuns

## 2. Parser

### Localização
- **Arquivo principal**: [`crates/souc/src/parser/mod.rs`](../../crates/souc/src/parser/mod.rs) (5,851 linhas)
- **Error handling**: [`crates/souc/src/parser/errors.rs`](../../crates/souc/src/parser/errors.rs)
- **Error recovery**: [`crates/souc/src/parser/recovery.rs`](../../crates/souc/src/parser/recovery.rs)

### Arquitetura

O parser é **recursive descent** com Pratt parsing para expressões.

#### Estrutura do Parser

```rust
// crates/souc/src/parser/mod.rs:84
pub struct Parser<'a> {
    tokens: &'a [Token],
    pos: usize,
    id_gen: IdGenerator,
    /// Para desambiguação de struct literals
    allow_struct_literals: bool,
    /// Mapeamento NodeId → Span para erros
    node_spans: HashMap<NodeId, Span>,
    /// Pending `>` para splitting de `>>`
    pending_gt: bool,
    source: &'a str,
}
```

#### Função Principal de Parsing

```rust
// crates/souc/src/parser/mod.rs:71
pub fn parse(tokens: &[Token], source: &str) -> Result<Ast> {
    let mut parser = Parser::with_source(tokens, source);
    parser.parse_program()
}

// crates/souc/src/parser/mod.rs:364
fn parse_program(&mut self) -> Result<Ast> {
    let mut items = Vec::new();

    // Coleta doc comments de nível arquivo
    let inner_doc = self.collect_inner_doc_comments();

    // Parse opcional de declaração de módulo
    let module_name = if self.at(TokenKind::Module) {
        // Peek ahead para distinguir: module foo; vs module foo { ... }
        let next = self.peek_n(1);
        let after_name = self.peek_n(2);

        if next == TokenKind::Ident && after_name == TokenKind::LBrace {
            None  // É inline module, não consumir
        } else {
            self.advance();
            let name = self.parse_path()?;
            if self.at(TokenKind::Semi) {
                self.advance();
            }
            Some(name)
        }
    } else {
        None
    };

    // Parse de items até EOF
    while !self.at(TokenKind::Eof) {
        items.push(self.parse_item()?);
    }

    Ok(Ast {
        module_name,
        items,
        inner_doc,
        node_spans: self.node_spans.clone(),
    })
}
```

### Parsing de Items

```rust
// crates/souc/src/parser/mod.rs:412
pub fn parse_item(&mut self) -> Result<Item> {
    let doc = self.collect_doc_comments();

    // Macro invocation: identifier! ou keyword!
    if self.can_be_macro_name() && self.peek_n(1) == TokenKind::Bang {
        return self.parse_macro_invocation();
    }

    let attributes = self.parse_item_attributes()?;
    let visibility = self.parse_visibility();
    let modifiers = self.parse_modifiers();

    match self.peek() {
        TokenKind::Fn | TokenKind::Kernel => self.parse_fn(visibility, modifiers, attributes, doc),
        TokenKind::Struct => self.parse_struct(visibility, modifiers, doc),
        TokenKind::Enum => self.parse_enum(visibility, modifiers, doc),
        TokenKind::Trait => self.parse_trait(visibility, modifiers, doc),
        TokenKind::Effect => self.parse_effect(visibility, doc),
        TokenKind::Handler => self.parse_handler(visibility, doc),
        TokenKind::Ode => self.parse_ode_def(visibility),
        TokenKind::Pde => self.parse_pde_def(visibility),
        TokenKind::Causal => self.parse_causal_model_def(visibility),
        // ... etc
    }
}
```

### AST (Abstract Syntax Tree)

### Localização
- **Definição**: [`crates/souc/src/ast/mod.rs`](../../crates/souc/src/ast/mod.rs) (2,003 linhas)

#### Estrutura Raiz

```rust
// crates/souc/src/ast/mod.rs:113
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Ast {
    pub module_name: Option<Path>,
    pub items: Vec<Item>,
    pub inner_doc: Option<String>,
    #[serde(skip)]
    pub node_spans: HashMap<NodeId, Span>,
}
```

#### Top-level Items

```rust
// crates/souc/src/ast/mod.rs:157
pub enum Item {
    Function(FnDef),
    Struct(StructDef),
    Enum(EnumDef),
    Trait(TraitDef),
    Impl(ImplDef),
    TypeAlias(TypeAliasDef),
    Effect(EffectDef),
    Handler(HandlerDef),
    Import(ImportDef),
    Export(ExportDef),
    Extern(ExternBlock),
    Global(GlobalDef),
    MacroInvocation(MacroInvocation),
    // Domínio científico
    OntologyImport(OntologyImportDef),
    AlignDecl(AlignDef),
    OdeDef(OdeDef),
    PdeDef(PdeDef),
    CausalModel(CausalModelDef),
    Unit(UnitDef),
    Module(ModuleDef),
}
```

#### Function Definition

```rust
// crates/souc/src/ast/mod.rs:241
pub struct FnDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: FnModifiers,
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generics: Generics,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub effects: Vec<EffectRef>,
    pub where_clause: Vec<WherePredicate>,
    pub body: Block,
    pub doc: Option<String>,
    pub span: Span,
}

pub struct FnModifiers {
    pub is_async: bool,
    pub is_unsafe: bool,
    pub is_kernel: bool,  // GPU kernel
    pub abi: Option<Abi>, // extern "C"
}
```

#### Struct Definition

```rust
// crates/souc/src/ast/mod.rs:274
pub struct StructDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub modifiers: TypeModifiers,
    pub attributes: Vec<Attribute>,
    pub name: String,
    pub generics: Generics,
    pub where_clause: Vec<WherePredicate>,
    pub fields: Vec<FieldDef>,
    pub doc: Option<String>,
    pub span: Span,
}

pub struct TypeModifiers {
    pub linear: bool,
    pub affine: bool,
}
```

#### Scientific DSL Support

```rust
// crates/souc/src/ast/mod.rs:192
pub struct UnitDef {
    pub id: NodeId,
    pub visibility: Visibility,
    pub name: String,
    pub definition: Option<UnitDefExpr>,  // None = base dimension
    pub doc: Option<String>,
    pub span: Span,
}

pub enum UnitDefExpr {
    Named(String),                       // kg, m, s
    Scale(f64, Box<UnitDefExpr>),        // 0.001 * g
    Product(Box<UnitDefExpr>, Box<UnitDefExpr>),  // kg * m
    Quotient(Box<UnitDefExpr>, Box<UnitDefExpr>), // m / s
    Power(Box<UnitDefExpr>, i8),         // m^2
}
```

### Sistema de Atributos

```rust
// crates/souc/src/ast/mod.rs:12
pub struct Attribute {
    pub id: NodeId,
    pub name: String,  // "compat", "derive", "cfg"
    pub args: AttributeArgs,
    pub span: Span,
}

pub enum AttributeArgs {
    Empty,
    Value(AttributeValue),
    Named(Vec<(String, AttributeValue)>),
    List(Vec<AttributeValue>),
}

// Exemplo de uso:
#[compat(threshold = 0.15)]
#[inline]
#[repr(C)]
pub fn foo() {}
```

### Expressões

A AST suporta 100+ variantes de expressões (ver [`crates/souc/src/ast/mod.rs`](../../crates/souc/src/ast/mod.rs:1423)):

- Literais: `i32`, `f64`, `string`, `char`, `bool`
- Compostas: arrays, tuples, structs, enums
- Controle: `if`, `match`, `loop`, `for`, `while`
- Funções: calls, closures, method calls
- Operadores: aritméticos, lógicos, bits, comparação
- Especiais: `Knowledge<T>`, `Dual<T>`, `do effects`

### Error Recovery

O parser inclui estratégias de recovery limitadas:

```rust
// crates/souc/src/parser/recovery.rs
// Estratégias:
// - Sincronização em boundaries de statements
// - Inserção de tokens esperados
// - Sugestões para erros comuns (e.g., &mut → &!)
```

### Desambiguations Especiais

#### Struct Literals vs Blocks
```sio
// Problema: Ident { ... } pode ser struct literal ou block
match x {
    Ident { name } => ...  // parse_expr() vs parse_block()
}

// Solução: Parser mantém allow_struct_literals flag
// que é desabilitado em contextos como match arms
```

#### Generics Aninhados
```sio
// Problema: Option<Box<T>> tem Tokens: Option < Box < T > >>
//          onde >> é um único token

// Solução: Parser mantém pending_gt flag
// Quando encontra Shr (>>), consome primeiro > e marca pending
```

## Fluxo de Parsing

```
parse(source)
    ↓
lexer::lex(source) → Vec<Token>
    ↓
Parser::new(tokens)
    ↓
Parser::parse_program()
    ├─ parse_module_declaration?
    ├─ loop: parse_item() until Eof
    │   ├─ parse_doc_comments
    │   ├─ parse_attributes
    │   ├─ parse_visibility
    │   ├─ parse_modifiers
    │   └─ dispatch to parse_fn/parse_struct/etc
    ↓
Result<Ast>
```

## Métricas

| Componente | Linhas | Tokens/Items |
|------------|--------|--------------|
| Lexer (tokens.rs) | 799 | 150+ TokenKinds |
| Lexer (mod.rs) | 239 | lex(), ~20 tests |
| Parser (mod.rs) | 5,851 | 100+ parsing functions |
| AST (mod.rs) | 2,003 | 100+ expr variants |

## Testes

O parser inclui testes abrangentes:

```rust
// crates/souc/src/lexer/mod.rs
#[test]
fn test_lex_simple() { /* ... */ }
#[test]
fn test_lex_keywords() { /* ... */ }
#[test]
fn test_lex_literals() { /* ... */ }
#[test]
fn test_lex_units() { /* ... */ }
#[test]
fn test_lex_effects() { /* ... */ }
#[test]
fn test_lex_comments() { /* ... */ }
#[test]
fn test_lex_doc_comments() { /* ... */ }

// crates/souc/src/parser/tests/
// Testes de parsing para cada construct da linguagem
```

## Próximos Passos

1. **Type Checker** → Verificar tipos, inferência,unificação
2. **Resolver** → Resolução de nomes e imports
3. **Effect Checker** → Verificação de efeitos
4. **HIR Lowering** → Conversão para High-Level IR

## Referências

- [`compiler/src/lexer/mod.rs`](../../crates/souc/src/lexer/mod.rs)
- [`compiler/src/lexer/tokens.rs`](../../crates/souc/src/lexer/tokens.rs)
- [`compiler/src/parser/mod.rs`](../../crates/souc/src/parser/mod.rs)
- [`compiler/src/ast/mod.rs`](../../crates/souc/src/ast/mod.rs)
- [Logos Documentation](https://docs.rs/logos/)
