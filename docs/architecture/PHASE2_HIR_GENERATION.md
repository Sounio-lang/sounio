<!-- docs:meta
topic_id: repo.docs.architecture.phase2-hir-generation
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.phase2-hir-generation
-->

# Phase 2: HIR Generation - Self-Hosted Sedenion Compiler

## Overview

Phase 2 converts type-checked AST → **High-Level Intermediate Representation (HIR)**.

**Goal**: Create a typed, name-resolved, semantically-analyzed representation suitable for optimization and code generation.

**Inputs**: Phase 0 (AST) + Phase 1 (roundtrip tests) + Type Checker results
**Outputs**: HIR nodes, control flow graphs, scope information
**Dependencies**: Phase 1 tests passing, sedenion_layer compilation working

---

## Architecture

```
Type-Checked AST (from Rust compiler's check/)
       ↓
   Name Resolution (resolve imports, local scopes)
       ↓
   Type Annotation (attach inferred types to nodes)
       ↓
   Desugaring (for..in → while loops, match guards → nested ifs)
       ↓
   Control Flow Analysis (build CFG, identify dominators)
       ↓
   HIR Construction (create intermediate representation)
       ↓
   Verification (type checking, scope checking, linearness)
       ↓
   HIR Output (ready for optimization/codegen)
```

---

## Phase 2 Components (to implement in Sounio)

### 1. Name Resolution Pass (`stdlib/compiler/transform/name_resolution.sio`)

**Purpose**: Resolve all identifier references to definitions.

**Key Functions**:
```sio
// Symbol table for a scope
type SymbolTable = {
    parent: Option<&SymbolTable>,
    bindings: Map<str, (NodeId, HirType)>,
    imports: Map<str, ModulePath>
}

// Build scope hierarchy from AST
fn build_symbol_tables(ast: &SedenionAST) -> ScopeMap

// Resolve a reference to its definition
fn resolve_identifier(name: str, scope: &SymbolTable) -> Result<(NodeId, HirType), ResolutionError>

// Validate no circular imports
fn check_import_acyclicity(imports: &[Import]) -> Result<(), CycleError>
```

**Tests**:
- Local variable shadowing
- Module imports with aliasing
- Function closure captures
- Generic type parameter resolution

---

### 2. Type Annotation Pass (`stdlib/compiler/transform/type_annotation.sio`)

**Purpose**: Attach resolved types to all AST nodes.

**Key Functions**:
```sio
// Annotated expression with type
type TypedExpr = {
    node_id: NodeId,
    expr: Expr,
    ty: HirType,
    effects: Vec<Effect>
}

// Pass types down through AST
fn annotate_expressions(ast: &SedenionAST, type_context: &TypeContext) -> Vec<TypedExpr>

// Infer types for inferrable positions
fn infer_missing_types(expr: &Expr, context: &TypeContext) -> Result<HirType, TypeError>

// Validate type consistency in subtrees
fn validate_type_tree(expr: &TypedExpr) -> Result<(), TypeError>
```

**Tests**:
- Bidirectional type checking
- Polymorphic function instantiation
- Unit type inference
- Refinement type witnesses

---

### 3. Desugaring Pass (`stdlib/compiler/transform/desugar.sio`)

**Purpose**: Convert high-level syntax to primitives.

**Examples**:
- `for x in xs { ... }` → `{var i = 0; while i < len(xs) { let x = xs[i]; ... }}`
- `match x { Some(v) if v > 0 => ... }` → nested if/match
- `array.map(f)` → explicit loop (if no iterators)
- `x?.foo()` → `match x { Ok(v) => v.foo(), Err(e) => return Err(e) }`

**Key Functions**:
```sio
// Desugar one construct at a time
fn desugar_for_loop(for_stmt: &ForStmt) -> Block

fn desugar_match_guard(arm: &MatchArm) -> MatchArm

fn desugar_optional_chaining(expr: &Expr) -> Expr

// Recursively desugar entire tree
fn desugar_tree(expr: &Expr) -> Expr
```

**Tests**:
- For loop with early breaks
- Match with multiple guards
- Nested desugaring
- Control flow preservation

---

### 4. Control Flow Analysis Pass (`stdlib/compiler/transform/cfg_builder.sio`)

**Purpose**: Build control flow graph (CFG) for function bodies.

**Structures**:
```sio
// Basic block in CFG
type BasicBlock = {
    id: BlockId,
    instructions: Vec<Instruction>,
    terminator: Terminator,
    predecessors: Vec<BlockId>,
    successors: Vec<BlockId>
}

// Terminator: return, branch, or unconditional jump
type Terminator =
    | Return(Option<Expr>)
    | Branch(Expr, BlockId, BlockId)  // if cond then block1 else block2
    | Unreachable

// Full control flow graph
type ControlFlowGraph = {
    entry_block: BlockId,
    blocks: Map<BlockId, BasicBlock>,
    dominators: Map<BlockId, BlockId>,  // immediate dominator
    post_dominators: Map<BlockId, BlockId>
}
```

**Key Functions**:
```sio
// Convert function body to CFG
fn build_cfg(body: &Block) -> Result<ControlFlowGraph, CfgError>

// Compute dominance relation (iterative fixed-point)
fn compute_dominators(cfg: &ControlFlowGraph) -> Map<BlockId, BlockId>

// Find loops and natural loops
fn find_loops(cfg: &ControlFlowGraph) -> Vec<Loop>

// Verify CFG properties (no orphaned blocks, all paths return)
fn verify_cfg(cfg: &ControlFlowGraph) -> Result<(), CfgError>
```

**Tests**:
- Linear sequence (no branches)
- If-then-else
- Nested loops
- Unreachable code detection
- Return path analysis

---

### 5. HIR Constructor (`stdlib/compiler/transform/hir_constructor.sio`)

**Purpose**: Build final HIR representation from desugared + annotated AST.

**Key Functions**:
```sio
// Convert top-level items
fn hir_from_item(item: &Item, name_table: &SymbolTable) -> HirItem

// Convert expressions to HIR expressions
fn hir_from_expr(expr: &TypedExpr, scope: &Scope) -> HirExpr

// Convert statements to HIR statements
fn hir_from_stmt(stmt: &Stmt, scope: &Scope) -> Vec<HirStmt>

// Full AST → HIR conversion
fn lower_to_hir(ast: &SedenionAST, type_context: &TypeContext) -> Result<Hir, HirError>
```

**Tests**:
- Item lowering (functions, structs, traits)
- Expression lowering (literals, operators, calls)
- Statement lowering (let bindings, assignments)
- Effect annotation propagation

---

### 6. HIR Verification Pass (`stdlib/compiler/transform/hir_verify.sio`)

**Purpose**: Validate HIR invariants before passing to next phase.

**Checks**:
- All names resolved (no dangling references)
- All expressions typed
- Type consistency (no mismatches)
- Linear types used linearly (first pass only warns)
- No unreachable code (conservative check)
- All paths in value-returning functions return

**Key Functions**:
```sio
// Verify single HIR item
fn verify_hir_item(item: &HirItem) -> Result<VerifyReport, HirError>

// Verify entire module
fn verify_hir(hir: &Hir) -> Result<VerifyReport, HirError>

// Report contains stats and warnings
type VerifyReport = {
    errors: Vec<HirError>,
    warnings: Vec<HirWarning>,
    stats: VerifyStats
}

type VerifyStats = {
    items_verified: i32,
    expressions_typed: i32,
    unreachable_blocks: i32,
    linear_type_issues: i32
}
```

**Tests**:
- All basic checks
- Error recovery
- Warning categorization

---

## Implementation Plan

### Timeline

| Phase | Task | Files | Est. LOC | Deps |
|-------|------|-------|---------|------|
| 2.1 | Name Resolution | name_resolution.sio | 250 | Phase 1 |
| 2.2 | Type Annotation | type_annotation.sio | 300 | 2.1 |
| 2.3 | Desugaring | desugar.sio | 350 | 2.2 |
| 2.4 | CFG Builder | cfg_builder.sio | 400 | 2.3 |
| 2.5 | HIR Constructor | hir_constructor.sio | 400 | 2.4 |
| 2.6 | HIR Verification | hir_verify.sio | 250 | 2.5 |
| 2.7 | Integration Tests | hir_integration_test.sio | 200 | 2.6 |

**Total**: ~2,000 lines of Sounio code

### Success Criteria

✅ All 6 components compile without errors
✅ 50+ unit tests (7-10 per component) pass
✅ Integration test: full sedenion_test.sio → HIR roundtrip succeeds
✅ Benchmark: <3 second type-check+HIR for 1000-line module
✅ All outputs verified (no dangling references, types consistent)

---

## Testing Strategy

### Unit Tests (per component)

**name_resolution_test.sio**:
- Local scope lookup
- Import resolution
- Shadowing
- Cycle detection

**type_annotation_test.sio**:
- Monomorphization
- Polymorphic generics
- Type inference
- Bidirectional checking

**desugar_test.sio**:
- For-loop expansion
- Match guards
- Optional chaining
- Control flow preservation

**cfg_test.sio**:
- Linear sequence CFG
- If-else CFG
- Loop CFG
- Unreachable code

**hir_constructor_test.sio**:
- Function lowering
- Struct lowering
- Expression lowering

**hir_verify_test.sio**:
- Name resolution check
- Type consistency check
- Linear type tracking

### Integration Test

**hir_integration_test.sio**:
```sio
// Test: AST → HIR roundtrip
fn test_hir_roundtrip() {
    let ast = parse_sedenion_file("examples/test.sio");
    let hir = lower_to_hir(ast);
    verify_hir(hir);
    // Verify all nodes reachable, types correct
}

// Test: Complex module
fn test_complex_module_hir() {
    // Load sedenion_test.sio, convert to HIR, verify
}
```

---

## Known Challenges

1. **Type inference performance**: Large polymorphic functions may slow bidirectional checking. Mitigation: cache type signatures, memoize inference.

2. **Circular module dependencies**: Import cycle detection needed. Use topological sort + error reporting.

3. **Linear type tracking**: Must track which values are consumed/dropped. Initially conservative (warn, don't error).

4. **Control flow in desugaring**: Ensuring desugared code maintains original semantics (esp. early returns, breaks).

---

## Integration with Existing Compiler

Phase 2 feeds into:
- **Phase 3** (epistemic tracking): Annotate confidence on HIR nodes
- **Phase 4** (partial eval): Use HIR for specialization
- **Phase 5** (bootstrap verify): Cross-validate HIR

Reuses from Phase 1:
- Sedenion AST encoding (for roundtrip tests)
- Checksums for output validation

---

## Next Steps After Phase 2

Once HIR generation is solid:
- **Phase 6**: MIR lowering (SSA construction, instruction selection)
- **Phase 7**: Optimization passes (constant prop, dead code elim, inlining)
- **Phase 8**: Backend selection (Cranelift/LLVM/native)
- **Phase 9**: Full bootstrap (stage-0/1/2 verification)

---

## References

- CLAUDE.md: Sounio syntax and language limitations
- COMPILER_ARCHITECTURE_OVERVIEW.md: Full pipeline
- Phase 1 design: Sedenion AST encoding + roundtrip tests
- Phase 3+ design: TBD based on Phase 2 completion
