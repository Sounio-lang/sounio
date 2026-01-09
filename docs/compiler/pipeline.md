# Compilation Pipeline

This document describes the complete compilation pipeline in Sounio, from source code to executable output.

## Pipeline Overview

```
Source (.sio)
     |
     | lexer::lex()
     v
Token Stream (Vec<Token>)
     |
     | parser::parse()
     v
AST (Abstract Syntax Tree)
     |
     | check::check()
     v
HIR (High-level IR)
     |
     | hlir::lower()
     v
HLIR (SSA-based IR)
     |
     | sir::lower() [optional]
     v
SIR (Scientific IR)
     |
     | codegen::*
     v
Machine Code / PTX / SPIR-V
```

## Data Flow Diagram

```
+-------------+     +-------------+     +-------------+
|   Source    | --> |    Lexer    | --> |   Tokens    |
|    .sio     |     |   (Logos)   |     |  Vec<Token> |
+-------------+     +-------------+     +-------------+
                                              |
                                              v
+-------------+     +-------------+     +-------------+
|   Errors    | <-- |   Parser    | <-- |   Tokens    |
|  (miette)   |     | (Recursive  |     |             |
+-------------+     |  Descent)   |     +-------------+
                    +-------------+
                          |
                          v
                    +-------------+
                    |     AST     |
                    | (Untyped)   |
                    +-------------+
                          |
                          v
+-------------+     +-------------+
| Type Errors | <-- | TypeChecker |
|             |     | (Bidirec-   |
+-------------+     |  tional)    |
                    +-------------+
                          |
                          v
                    +-------------+
                    |     HIR     |
                    |  (Typed)    |
                    +-------------+
                          |
                          v
                    +-------------+
                    | HLIR Lower  |
                    |  (SSA)      |
                    +-------------+
                          |
                          v
                    +-------------+
                    |    HLIR     |
                    | (SSA Form)  |
                    +-------------+
                          |
            +-------------+-------------+
            |             |             |
            v             v             v
      +---------+   +---------+   +---------+
      | LLVM    |   |Cranelift|   |   GPU   |
      | Backend |   |   JIT   |   | Backend |
      +---------+   +---------+   +---------+
            |             |             |
            v             v             v
      +---------+   +---------+   +---------+
      | Native  |   | JIT     |   |PTX/SPIRV|
      |  Code   |   | Execute |   |  /MSL   |
      +---------+   +---------+   +---------+
```

## Stage 1: Lexical Analysis (Lexer)

**Location**: `compiler/src/lexer/`

**Input**: Source code as a string (`&str`)

**Output**: `Vec<Token>`

**Entry Point**: `lexer::lex(source: &str) -> Result<Vec<Token>>`

The lexer uses the [Logos](https://github.com/maciejhirsz/logos) library for high-performance tokenization. It transforms raw source text into a stream of tokens.

### Key Characteristics

- Zero-copy tokenization where possible
- Handles doc comments (`///`, `//!`, `/** */`, `/*! */`)
- Recognizes unit literals (e.g., `500_mg`, `10.5_mL`)
- Supports Unicode operators (`+-` for plus-minus, `\partial` for partial derivative)
- Skips regular comments but captures doc comments as tokens

### Token Structure

```rust
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
    pub text: String,
}
```

See [frontend/lexer.md](frontend/lexer.md) for complete documentation.

## Stage 2: Parsing

**Location**: `compiler/src/parser/`

**Input**: `Vec<Token>`

**Output**: `Ast`

**Entry Point**: `parser::parse(tokens: &[Token], source: &str) -> Result<Ast>`

The parser uses a combination of recursive descent for statements/items and Pratt parsing for expressions with proper precedence handling.

### Key Characteristics

- Recursive descent for top-level items and statements
- Pratt parsing for expressions (handles operator precedence)
- Error recovery with synchronization points
- Newline-aware parsing (prevents `(...)` on new line from being parsed as call)
- Handles `>>` token splitting for nested generics (`Option<Box<T>>`)

### AST Structure

```rust
pub struct Ast {
    pub module_name: Option<Path>,
    pub items: Vec<Item>,
    pub node_spans: HashMap<NodeId, Span>,
}
```

See [frontend/parser.md](frontend/parser.md) and [frontend/ast.md](frontend/ast.md) for details.

## Stage 3: Type Checking

**Location**: `compiler/src/check/`

**Input**: `Ast`

**Output**: `Hir`

**Entry Point**: `check::check(ast: &Ast) -> Result<Hir>`

The type checker performs semantic analysis and produces a typed intermediate representation (HIR).

### Responsibilities

1. **Name Resolution**: Resolve identifiers to their definitions
2. **Type Inference**: Bidirectional type inference with unification
3. **Effect Checking**: Verify effect annotations and infer effect sets
4. **Ownership Analysis**: Verify linear/affine type usage
5. **Unit Checking**: Dimensional analysis for quantities
6. **Ontology Validation**: Verify ontology term references
7. **Epistemic Constraints**: Check Knowledge<T> type constraints

### Multi-Pass Approach

```
Pass 1: Collect type definitions, ontology prefixes, alignments
Pass 2: Check for circular types, infinite-size structs
Pass 3: Type check items (functions, structs, impls, etc.)
Pass 4: Solve type constraints via unification
Pass 5: Check for unused variables, unreported warnings
```

See [middle/type-checking.md](middle/type-checking.md) for complete documentation.

## Stage 4: HIR (High-level IR)

**Location**: `compiler/src/hir/`

The HIR is the typed AST produced by the type checker. It contains:

- Resolved types for all expressions
- Resolved names (no more path resolution needed)
- Desugared constructs
- Ownership and borrowing information

### Key Types

```rust
pub struct Hir {
    pub items: Vec<HirItem>,
    pub externs: Vec<HirExternBlock>,
}

pub enum HirItem {
    Function(HirFn),
    Struct(HirStruct),
    Enum(HirEnum),
    Trait(HirTrait),
    Impl(HirImpl),
    TypeAlias(HirTypeAlias),
    Effect(HirEffect),
    Handler(HirHandler),
    Global(HirGlobal),
}
```

## Stage 5: HLIR Lowering

**Location**: `compiler/src/hlir/`

**Input**: `Hir`

**Output**: `HlirModule`

**Entry Point**: `hlir::lower(hir: &Hir) -> HlirModule`

HLIR is an SSA-based intermediate representation suitable for optimization and code generation.

### Key Characteristics

- Static Single Assignment (SSA) form
- Basic blocks with explicit control flow
- Explicit memory operations
- Type-safe operations

### HLIR Structure

```rust
pub struct HlirModule {
    pub name: String,
    pub functions: Vec<HlirFunction>,
    pub structs: Vec<HlirStruct>,
    pub globals: Vec<HlirGlobal>,
}

pub struct HlirFunction {
    pub name: String,
    pub params: Vec<HlirParam>,
    pub return_type: HlirType,
    pub blocks: Vec<HlirBlock>,
    pub entry_block: BlockId,
}
```

## Stage 6: SIR (Scientific IR) - Optional

**Location**: `compiler/src/sir/`

**Input**: `HlirModule`

**Output**: `SirModule`

SIR is a domain-aware low-level IR designed specifically for Sounio's unique features. It preserves domain-specific information that enables specialized optimizations.

### Key Characteristics

- **Epistemic awareness**: Confidence and provenance flow through the IR
- **Numerical semantics**: IEEE 754 guarantees, precision requirements
- **Probability primitives**: First-class distribution sampling
- **Scientific patterns**: ODE solver steps, compartment models

### Architecture

```
HLIR (Sounio types, high-level ops)
  | lowering
  v
SIR (machine types, domain metadata)
  | optimization passes
  v
SIR (optimized)
  | code emission
  v
Machine Code (x86-64, ARM64)
```

### Example SIR Output

```
; HLIR input:
;   let x: Epistemic<f64> = 0.5 ~ confidence(0.9)
;   let y = x * 2.0

; SIR output:
define @main() -> !sir.void {
entry:
  %x.val = sir.const.f64 0.5
  %x.conf = sir.const.f64 0.9
  %two = sir.const.f64 2.0
  %y.val = sir.mul.f64 %x.val, %two
  %y.conf = sir.epistemic.propagate.mul %x.conf, 1.0
  sir.return.void
}
```

## Stage 7: Code Generation

**Location**: `compiler/src/codegen/`

The codegen stage transforms HLIR (or SIR) into executable code using one of several backends.

### Available Backends

| Backend | Feature Flag | Use Case |
|---------|--------------|----------|
| Cranelift | `jit` | Fast JIT compilation, development |
| LLVM | `llvm` | Optimized AOT compilation |
| GPU (PTX) | `gpu` | NVIDIA CUDA kernels |
| GPU (SPIR-V) | `gpu` | Vulkan/OpenCL kernels |
| GPU (Metal) | `gpu` | Apple Silicon GPUs |

### Backend Selection

```rust
pub enum Backend {
    Native,    // SIR direct emission
    LLVM,      // LLVM for optimized native code
    Cranelift, // Fast JIT compilation
    GPU,       // GPU compute kernels
}
```

See [backend/codegen.md](backend/codegen.md) for complete documentation.

## Special Processing

### Effect Handling

Effects are tracked through the entire pipeline:

1. **Parsing**: Effect annotations (`with IO, Mut`) are captured
2. **Type Checking**: Effect sets are inferred and verified
3. **HIR**: Effects are stored in function signatures
4. **HLIR**: Effect operations become explicit calls
5. **Codegen**: Effect handlers are generated (CPS or direct)

### Epistemic Types

Knowledge<T> types with uncertainty propagation:

1. **Parsing**: Knowledge type syntax is parsed
2. **Type Checking**: Epistemic constraints are verified
3. **HIR**: Epistemic metadata is preserved
4. **SIR**: Shadow registers for confidence values
5. **GPU**: Epistemic-aware PTX emission with shadow registers

### GPU Kernel Compilation

For `kernel` functions:

```
HLIR
  | hlir_to_gpu::lower()
  v
GpuIR (GPU-specific IR)
  | PtxCodegen / SpirvCodegen / MetalCodegen
  v
PTX / SPIR-V / MSL
  | Driver
  v
GPU Execution
```

## Error Handling

Errors at each stage use the [miette](https://docs.rs/miette) library for rich diagnostics:

```rust
// Lexer errors
miette::miette!("Unexpected character at position {}", pos)

// Parser errors
errors::ParserError::UnexpectedToken { span, expected, found, context }

// Type errors
TypeError { message, span, code }
```

All errors include source spans for precise error location reporting.

## Entry Points

The main entry points in `compiler/src/lib.rs`:

```rust
// Full compilation (requires jit or llvm feature)
pub fn compile(source: &str) -> miette::Result<Vec<u8>>

// Type checking only
pub fn typecheck(source: &str) -> miette::Result<Hir>

// Parse only
pub fn parse(source: &str) -> miette::Result<Ast>

// Interpretation
pub fn interpret(source: &str) -> miette::Result<interp::Value>

// GPU compilation
pub fn compile_to_gpu(source: &str, sm_version: (u32, u32)) -> miette::Result<String>
pub fn compile_to_gpu_epistemic(source: &str, sm_version: (u32, u32)) -> miette::Result<String>
```
