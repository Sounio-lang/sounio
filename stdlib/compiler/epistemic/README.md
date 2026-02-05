# Compiler Epistemic Modules - Phase 3

## Status: Specification Phase

These modules represent Phase 3 of the epistemic system for the Sounio compiler:
**Epistemic Confidence Tracking During Type Checking**.

## Architecture

### Tier 1 - Foundation (3 modules)
1. **confidence_metadata.sio** (250 lines)
   - ConfidenceLevel enum (5 discrete levels)
   - InferenceSource enum (8 source types)
   - TypeConfidenceMetadata, ConstraintConfidence
   - Core metadata types for tracking epistemic state

2. **type_confidence.sio** (320 lines)
   - TypeWithConfidence wrapper
   - SubstitutionConfidence, UnificationConfidence
   - Generalization/Instantiation confidence tracking
   - Type-level operations with confidence

3. **propagation.sio** (400 lines)
   - PropagationContext for global tracking
   - Constraint propagation to fixpoint
   - Expression inference with confidence
   - Pattern matching confidence

### Tier 2 - Integration (4 modules)
4. annotation.sio - User-facing confidence annotations
5. attestation.sio - Proof/evidence attachment
6. epistemic_hir.sio - HIR with epistemic metadata
7. verify.sio + tests - Verification and testing

## Current Implementation Status

### Parser Limitations Discovered

During LLM-assisted code generation (Session 3, 2026-02-05), we discovered:

1. **Tuple structs not supported**: `pub struct NodeId(u32)` fails to parse
   - Error: "Expected {, found ("
   - Workaround: Use regular structs or import from compiler internals

2. **Impl blocks incomplete**: `impl Foo { ... }` causes parse errors
   - Error: "Expected ,, found :"
   - Affects method definitions like `pub fn foo(self: &Foo) -> i32`
   - Note: stdlib/epistemic/knowledge.sio also doesn't compile

3. **Doc comments**: `///` triggers "Expected identifier, found DocCommentOuter"

### What Works (from tests/run-pass/)
- Basic structs with fields: `struct Foo { x: i32 }`
- Enums with variants: `enum Color { Red, Blue }`
- Enum struct variants: `enum Result { Ok { value: i32 } }`
- Functions with effects: `fn main() with IO { ... }`
- Pattern matching, let bindings, literals

### Recommendations

1. **Short term**: Implement these modules in Rust as compiler built-ins
   - `crates/souc/src/epistemic/confidence.rs` (metadata types)
   - `crates/souc/src/check/confidence.rs` (type checker integration)
   - Reference these .sio files as specification

2. **Medium term**: Complete parser support for:
   - Impl blocks with methods
   - Tuple structs
   - Doc comments (/// and //!)

3. **Long term**: Port to Sounio once parser is feature-complete

## LLM Offloading Experiment (Session 3)

**Method**: Used mcp__minimax__offload_generate_code with 3 providers:
- minimax 2.1 (confidence_metadata) - structural code
- Grok-4.1-fast (type_confidence, propagation) - algorithmic code

**Results**:
- ✅ Generated valid scaffolds (~1000 lines total)
- ✅ Correct structure and type signatures
- ⚠️ Truncated at max_tokens=4096 (all 3 files incomplete)
- ⚠️ Rust-isms present (String vs string, format!() macro, etc.)
- ❌ Generated code doesn't compile due to parser limitations (not LLM fault)

**Conclusion**: LLM offloading is viable for scaffold generation, but:
1. Need higher max_tokens (6000+) for complete modules
2. Specs must be detailed about Sounio vs Rust syntax
3. Claude review essential for cleanup
4. Parser feature gaps are the blocker, not LLM quality

## References

- CLAUDE.md - Sounio syntax reference
- stdlib/epistemic/knowledge.sio - Runtime Knowledge<T> type
- docs/LLM_PROGRAMMING_GUIDE.md - Language syntax guide
- .claude/memory/MEMORY.md - Session notes

## Integration Plan

Once parser supports impl blocks:

1. Port confidence_metadata.sio to Sounio
2. Integrate with crates/souc/src/check/rankn.rs (unification)
3. Add confidence tracking to TypeChecker::infer()
4. Emit warnings for low-confidence inferences
5. Add epistemic flags: --warn-low-confidence, --require-confidence=<level>

## Timeline

- Phase 3 Tier 1 specs: ✅ Complete (2026-02-05)
- Parser impl blocks: Pending
- Rust implementation: 2-3 days
- Sounio port: After parser complete
- Integration + tests: 1 week
