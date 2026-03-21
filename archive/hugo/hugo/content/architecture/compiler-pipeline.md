---
title: "Compiler Pipeline"
description: "Detailed walkthrough of Sounio's 8-stage compilation pipeline with data structures and code references."
---

## Compiler Pipeline

### Stage 1: Lexer

**File**: `compiler/src/lexer/mod.rs:14`

The lexer uses the **Logos** library for fast tokenization. Each token carries a `TokenKind`, source `Span`, and text content.

```rust
pub fn lex(source: &str) -> Result<Vec<Token>>
```

### Stage 2: Parser

**File**: `compiler/src/parser/mod.rs:19` (5,656 lines)

A **recursive descent** parser with explicit state management:

```rust
pub fn parse(tokens: &[Token], source: &str) -> Result<Ast>
```

The `Parser` struct maintains:
- **Token stream** with position tracking
- **`allow_struct_literals`** flag for disambiguating `Ident { ... }` (expression vs. block)
- **`pending_gt`** flag for splitting `>>` in nested generics
- **`node_spans`** for error reporting with source locations

Key parsing methods:
- `parse_program()` (line 291) --- Collects top-level items
- `parse_item()` (line 343) --- Dispatches to function, struct, enum, etc.
- `parse_expr()` (line 3688) --- Primary expression entry with precedence climbing
- `parse_type()` --- Type expression parsing with generic nesting

**Error recovery**: The parser detects Rust patterns (e.g., `&mut`) and suggests Sounio equivalents (`&!`).

### Stage 3: AST

**File**: `compiler/src/ast/mod.rs:113`

The `Ast` root structure:

```rust
pub struct Ast {
    pub module_name: Option<Path>,
    pub items: Vec<Item>,
    pub inner_doc: Option<String>,
    pub node_spans: HashMap<NodeId, Span>,
}
```

**Top-level items** (line 158) include:
- `Function(FnDef)`, `Struct(StructDef)`, `Enum(EnumDef)`
- `Trait(TraitDef)`, `Impl(ImplDef)`
- `Effect(EffectDef)`, `Handler(HandlerDef)`
- Domain-specific: `OdeDef`, `PdeDef`, `CausalModel`, `UnitDef`

The `Expr` enum (line 1423) has **100+ variants** covering all expression forms.

### Stage 4: Type Checker

**File**: `compiler/src/check/mod.rs:37` (7,153 lines)

```rust
pub fn check_ast(ast: &Ast) -> Result<Hir>
```

The type checker uses **bidirectional type inference** with constraint solving:

```rust
struct TypeChecker {
    env: TypeEnv,                    // Variable bindings with scope stack
    type_defs: HashMap<String, TypeDef>,
    effects: EffectInference,        // Algebraic effect inference
    units: UnitChecker,              // Dimensional analysis
    next_type_var: u32,              // Fresh type variable counter
    next_effect_var: u32,            // Effect variable for row polymorphism
    constraints: Vec<TypeConstraint>, // Unification constraints
    handler_effects: HashMap<String, String>,
    masked_effects: EffectSet,       // Effects handled in current scope
}
```

The type checker simultaneously performs:
1. **Type inference** (bidirectional with unification)
2. **Effect inference** (row-polymorphic effect sets)
3. **Unit checking** (dimensional analysis)
4. **Linearity checking** (linear/affine type enforcement)
5. **Ontology alignment** (semantic compatibility scoring)

### Stage 5: HIR (High-Level IR)

**File**: `compiler/src/hir/mod.rs:20`

```rust
pub struct Hir {
    pub items: Vec<HirItem>,
    pub externs: Vec<HirExternBlock>,
}
```

The HIR preserves high-level semantics:
- **`HirFn`**: Functions with typed bodies, effect annotations, and linearity markers
- **`HirStruct`**: Structs with `is_linear` and `is_affine` flags
- **`HirEnum`**: Enums with optional GADT return types
- **`HirFnType`**: Function signatures with explicit `effects: Vec<HirEffect>`

### Stage 6: HLIR (SSA-Based Low-Level IR)

**File**: `compiler/src/hlir/ir.rs:13`

```rust
pub struct HlirModule {
    pub functions: Vec<HlirFunction>,
    pub globals: Vec<HlirGlobal>,
    pub types: Vec<HlirTypeDef>,
}
```

The HLIR uses **Static Single Assignment (SSA)** form with explicit basic blocks:

```rust
pub struct HlirFunction {
    pub blocks: Vec<HlirBlock>,           // SSA basic blocks
    pub locals: HashMap<ValueId, HlirType>,
    pub is_kernel: bool,                  // GPU kernel flag
}

pub struct HlirBlock {
    pub id: BlockId,
    pub instructions: Vec<HlirInstr>,     // SSA operations
    pub terminator: HlirTerminator,       // Control flow
}
```

**Operations** (line 403): 30+ SSA operations including arithmetic, memory, aggregates, FFI, and effect operations (`PerformEffect`, `DispatchEffect`, `PushHandler`, `PopHandler`).

**Type system** (line 129): 40+ types including primitives, aggregates, linear algebra types (`Vec2`-`Vec4`, `Mat2`-`Mat4`, `Quat`), exotic types (`Octonion`, `Dual`), ML types (`QuatLinear`, `QuatConv2d`), and epistemic types (`Knowledge { inner, mode, epsilon_bound, provenance_id }`).

### Stage 7: SIR (Scientific IR)

**File**: `compiler/src/sir/mod.rs:19`

The SIR preserves domain-specific semantics that would be lost in LLVM IR:

```
HLIR (Sounio types, high-level ops)
  |  lowering
SIR (machine types, domain metadata)
  |  optimization passes
SIR (optimized)
  |  code emission
Machine Code (x86-64, ARM64)
```

SIR types include vector types (`f64x2`, `f64x4`, `f32x8`), scalar types with explicit alignment, and metadata tracking for epistemic values, physical units, and field-level annotations.

### Stage 8: Code Generation

**HIR to HLIR lowering** (`hlir/lower.rs:12`):
- Refinement types lowered to base type (e.g., `type Positive = { x: i32 | x > 0 }` becomes `i32`)
- Epistemic types encoded with `mode`, `epsilon_bound`, `provenance_id`
- Async functions transformed to state machines

**HLIR to machine code**:
- **LLVM path**: HLIR -> LLVM IR -> optimization passes -> object file -> linker
- **Cranelift path**: HLIR -> Cranelift IR -> JIT execution
- **Native path**: HLIR -> SIR -> x86-64 machine code -> ELF object -> linker
- **GPU path**: HLIR -> GPU IR -> PTX/SPIR-V/Metal -> GPU executable
