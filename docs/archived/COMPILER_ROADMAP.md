<!-- docs:meta
topic_id: repo.docs.archived.compiler-roadmap
authority: archived
audience: maintainers
last_validated: 2026-03-07
validated_by: A7
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.archived.compiler-roadmap
-->


<!-- docs:status-note:start -->
> Docs status: `archived`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# SOUNIO COMPILER — IMPLEMENTATION ROADMAP

## Visão Geral: Fases do Compilador

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SOUNIO COMPILER PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Source (.sio)                                                         │
│       │                                                                 │
│       ▼                                                                 │
│   ┌─────────┐     FASE 1                                               │
│   │  LEXER  │     Análise Léxica                                       │
│   └────┬────┘     tokens                                               │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐     FASE 2                                               │
│   │ PARSER  │     Análise Sintática                                    │
│   └────┬────┘     AST                                                  │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐     FASE 3                                               │
│   │ TYPECK  │     Análise Semântica                                    │
│   └────┬────┘     Typed AST                                            │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐     FASE 4                                               │
│   │   HIR   │     High-level IR                                        │
│   └────┬────┘                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐     FASE 5                                               │
│   │   MIR   │     Mid-level IR (SSA)                                   │
│   └────┬────┘                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────┐     FASE 6                                               │
│   │ CODEGEN │     Cranelift                                            │
│   └────┬────┘                                                          │
│        │                                                                │
│        ▼                                                                │
│   Binary (.exe)                                                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

# FASE 0: PROJECT BOOTSTRAP (Dia 1)

## Objetivo
Criar estrutura do projeto Rust com workspace.

## Arquivos a Criar

### `Cargo.toml` (Workspace Root)

```toml
[workspace]
resolver = "2"
members = [
    "compiler",
    "runtime",
]

[workspace.package]
version = "0.93.0"
edition = "2021"
authors = ["Demetrios Chiuratto Agourakis <agourakis82@gmail.com>"]
license = "MIT"
repository = "https://github.com/sounio-lang/sounio"

[workspace.dependencies]
# Shared dependencies
logos = "0.14"
chumsky = "0.9"
ariadne = "0.4"
cranelift = "0.112"
cranelift-module = "0.112"
cranelift-object = "0.112"
clap = { version = "4.5", features = ["derive"] }
thiserror = "2.0"
```

---

# FASE 1: LEXICAL ANALYSIS (Dias 2-4)

## Objetivo
Transformar source code em stream de tokens.

### Token Types

```rust
#[derive(Logos, Debug, Clone, PartialEq)]
pub enum TokenKind {
    // Keywords
    #[token("fn")] Fn,
    #[token("let")] Let,
    #[token("mut")] Mut,
    #[token("if")] If,
    #[token("else")] Else,
    #[token("match")] Match,
    #[token("while")] While,
    #[token("for")] For,
    #[token("in")] In,
    #[token("return")] Return,
    #[token("struct")] Struct,
    #[token("enum")] Enum,
    #[token("impl")] Impl,
    #[token("trait")] Trait,
    #[token("pub")] Pub,

    // Sounio-specific
    #[token("model")] Model,
    #[token("param")] Param,
    #[token("compartment")] Compartment,
    #[token("flow")] Flow,
    #[token("observe")] Observe,

    // Literals
    #[regex(r"[0-9][0-9_]*")] IntLit,
    #[regex(r"[0-9][0-9_]*\.[0-9][0-9_]*")] FloatLit,
    #[regex(r#""([^"\\]|\\.)*""#)] StringLit,

    // Identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")] Ident,

    // Operators
    #[token("+")] Plus,
    #[token("-")] Minus,
    #[token("*")] Star,
    #[token("/")] Slash,
    #[token("==")] EqEq,
    #[token("!=")] NotEq,
    #[token("->")] Arrow,
    // ...
}
```

---

# FASE 2: SYNTAX ANALYSIS (Dias 5-10)

## Objetivo
Transformar tokens em Abstract Syntax Tree (AST).

### AST Nodes

```rust
pub struct Module {
    pub name: String,
    pub items: Vec<Item>,
}

pub enum Item {
    Function(FunctionDef),
    Struct(StructDef),
    Enum(EnumDef),
    Model(ModelDef),
}

pub struct FunctionDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Option<Block>,
}

pub enum ExprKind {
    Literal(Literal),
    Ident(String),
    Binary { op: BinaryOp, left: Box<Expr>, right: Box<Expr> },
    Call { callee: Box<Expr>, args: Vec<Expr> },
    Knowledge { value: Box<Expr>, uncertainty: Box<Expr>, confidence: Option<Box<Expr>> },
    // ...
}
```

---

# FASE 3: TYPE SYSTEM (Semanas 1-2)

See `ROADMAP_v0.5.0_PART1.md` for details.

---

# FASE 4: HIR (Semanas 3-4)

High-level IR after type checking.

---

# FASE 5: MIR (Semanas 5-6)

Mid-level IR in SSA form.

---

# FASE 6: CODEGEN (Semanas 7-8)

Cranelift backend for native code generation.

---

## Estrutura Final

```
compiler/
├── Cargo.toml
└── src/
    ├── main.rs
    ├── lib.rs
    ├── lexer/
    │   ├── mod.rs
    │   └── token.rs
    ├── ast/
    │   ├── mod.rs
    │   ├── expr.rs
    │   ├── stmt.rs
    │   └── types.rs
    ├── parser/
    │   ├── mod.rs
    │   └── grammar.rs
    ├── typeck/
    │   ├── mod.rs
    │   ├── infer.rs
    │   └── unify.rs
    ├── hir/
    │   └── mod.rs
    ├── mir/
    │   └── mod.rs
    ├── codegen/
    │   ├── mod.rs
    │   └── cranelift.rs
    ├── diagnostics/
    │   ├── mod.rs
    │   └── errors.rs
    └── driver/
        ├── mod.rs
        └── session.rs
```

---

## Milestones

| Fase | Comando | Status |
|------|---------|--------|
| 0-1 | `souc dump-tokens file.sio` | 🎯 |
| 2 | `souc dump-ast file.sio` | 🎯 |
| 3 | `souc check file.sio` | 🎯 |
| 4-6 | `souc build file.sio` | 🎯 |
| 4-6 | `souc run file.sio` | 🎯 |

---

🏛️ SOUNIO — Compiler Implementation Roadmap
