<!-- docs:meta
topic_id: repo.docs.stdlib.stdlib-language-limitations
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.stdlib.stdlib-language-limitations
-->

# Stdlib Module Language Limitations

## Summary

Three critical library modules required for GitHub Issue #18 (multi-phase scientific computing workflow) have been designed but **cannot yet be implemented** in Sounio due to current language parser limitations.

These modules are essential for network science research:
- `stdlib/stats/effect_sizes.sio` - Statistical effect size calculations (Cliff's delta, Cohen's d, confidence intervals)
- `stdlib/data/csv_loader.sio` - CSV edge list loading and parsing
- `stdlib/graph/nulls/configuration.sio` - Configuration model null hypothesis generation

## Identified Limitations

### 1. Generic Type Parameters in Function Calls

**Current Status**: ❌ Not supported

```sio
// DOESN'T WORK:
var v = vec_new::<f64>()
var adj = vec_new::<Vec<usize>>()

// WORKAROUND NEEDED:
// Type inference or explicit wrapper functions
```

**Impact**: Cannot create vectors with type hints. Must rely on type inference from context.

**Affected Modules**: All three (effect_sizes, csv_loader, configuration)

---

### 2. Method Call Syntax with Type Parameters

**Current Status**: ❌ Not supported

```sio
// DOESN'T WORK:
let value = string_value.parse::<usize>()

// WORKAROUND NEEDED:
// Standalone parsing functions or manual parsing
pub fn parse_usize(s: &str) -> Result<usize, ParseError> {
    // Manual digit-by-digit parsing
}
```

**Impact**: Cannot use polymorphic methods. Must provide type-specific functions.

**Affected Modules**: csv_loader (needs to parse usize, f64 from strings)

---

### 3. Complex Nested Generic Return Types

**Current Status**: Partially working

```sio
// PROBLEMATIC:
pub fn configuration_model_replicates(
    adj: &[Vec<usize>],
    num_replicates: usize
) -> Vec<Vec<Vec<usize>>> with IO, Alloc {
    // Triple-nested Vec
}

// WORKS in function signatures but...
// May fail in complex struct definitions or method chaining
```

**Impact**: Can define return types but type inference breaks down in complex scenarios.

**Affected Modules**: configuration (multiple nested vector types)

---

## Proposed Solutions

### Option A: Language Enhancement (Compiler Work)

Enhance Sounio's parser and type inference to support:

1. **Generic type parameters in function calls**
   ```sio
   fn vec_new<T>() -> Vec<T>
   ```

2. **Method call syntax with type parameters**
   ```sio
   fn String.parse<T>(self) -> Result<T, ParseError>
   ```

3. **Polymorphic parsing**
   ```sio
   trait FromStr<T> {
       fn from_str(s: &str) -> Result<T, ParseError>
   }
   ```

**Effort**: Significant compiler work (parser, type checker, codegen)

**Timeline**: Phase 3+ work

---

### Option B: Stdlib Workaround (Library Work)

Rewrite modules to work within current language constraints:

#### csv_loader.sio - Type-Specific Parsing Functions

```sio
pub fn parse_usize_line(line: &str) -> Result<usize, String> with Alloc {
    // Manual digit-by-digit parsing without type parameters
    var result = 0
    for c in line.chars() {
        if c >= '0' && c <= '9' {
            result = result * 10 + (c as usize - '0' as usize)
        } else {
            return Err(string_from("Invalid digit"))
        }
    }
    Ok(result)
}

pub fn parse_f64_line(line: &str) -> Result<f64, String> with Alloc {
    // Manual float parsing
    // (More complex: handle exponent, fractional parts)
}
```

**Workaround**: Create separate functions for each type:
- `parse_edge_line_usize()`
- `parse_edge_line_f64()`

#### effect_sizes.sio - Keep as-is

Return types like `(f64, f64)` should work. Main issue is if the module tries to use generic parse methods.

#### configuration.sio - Simplify Type Nesting

Break complex types into named structs:
```sio
pub struct DegreeSequence {
    pub degrees: Vec<usize>,
    pub total: usize,
}

pub struct GraphAdjacency {
    pub nodes: Vec<Vec<usize>>,  // Instead of Vec<Vec<Vec<usize>>>
}
```

**Effort**: Medium (refactor each module, ~4-6 hours)

**Timeline**: Can be done now if needed

---

## Current Phase Blockers

- **GitHub Issue #17**: Phase 2 configuration model testing blocked until `stdlib/graph/nulls/configuration.sio` is available
- **GitHub Issue #18**: Multi-module composition testing blocked until all three modules compile
- **GitHub Issue #21**: Build system linking issues may also be relevant

---

## Recommendation

1. **Short-term (Now)**:
   - Document these limitations in compiler roadmap
   - Create simple working alternatives that demonstrate the pattern
   - File enhancement issues for parser improvements

2. **Medium-term (v0.70-v0.72)**:
   - Implement generic type parameter syntax
   - Add method call support with type hints
   - Implement trait-based polymorphism (FromStr, Into, etc.)

3. **Long-term (v1.0)**:
   - Full generic type system with bounds
   - Monomorphization in codegen
   - Optimization for generic code

---

## Files Affected

Created but not yet compilable:

1. `stdlib/stats/effect_sizes.sio` (~525 LOC)
   - Status: Tuple return types likely OK, but needs testing
   - Issue: May have parse method usage

2. `stdlib/data/csv_loader.sio` (~309 LOC)
   - Status: BLOCKED - requires `.parse::<T>()` method syntax
   - Workaround: Replace with type-specific parsing functions

3. `stdlib/graph/nulls/configuration.sio` (~423 LOC)
   - Status: BLOCKED - deeply nested generic types
   - Workaround: Use named struct intermediates

---

## Example Demonstration Programs

Created to show intended workflow (compile but can't run without stdlib):

1. `examples/stats/effect_sizes_demo.sio` - Shows API usage pattern
2. `examples/network/null_hypothesis_demo.sio` - Shows full research workflow

These can serve as reference implementations once the compiler limitations are addressed.

---

## See Also

- [MINIMUM_VIABLE_SOUNIO.md](../guide/MINIMUM_VIABLE_SOUNIO.md) - Current language capabilities
- [DEVELOPER_WORKFLOW.md](../contributor-guide/DEVELOPER_WORKFLOW.md) - Development guidelines
- GitHub Issue #18 - Stdlib Module Design RFC
- GitHub Issue #21 - Multi-module linking issues
