<!-- docs:meta
topic_id: repo.docs.architecture.mir-optimization-pases-documentation
authority: historical
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.architecture.mir-optimization-pases-documentation
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# MIR Optimization Passes - Technical Documentation

## Overview

This document provides comprehensive technical documentation for the MIR (Mid-level Intermediate Representation) optimization passes implemented in the Sounio compiler. The optimization pipeline provides a modular, extensible framework for improving generated code quality through various analysis and transformation passes.

## Architecture

### Core Components

```
HLIR → MIR → [Optimizer] → Cranelift → Native Code
```

The optimization pipeline operates on MIR, which provides:

- **SSA Form**: Single Static Assignment representation for register allocation
- **Basic Blocks**: Explicit control flow with blocks and terminators
- **Type Safety**: Type-checked operations and validations
- **Backend Independence**: Machine-independent optimizations

### Pass Manager System

The optimization framework is built around a trait-based system:

```rust
pub trait MIRPass {
    fn name(&self) -> &'static str;
    fn run_on_module(&self, module: &mut MirModule) -> Result<bool, String>;
    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String>;
    fn requires_ssa(&self) -> bool;
    fn preserves_ssa(&self) -> bool;
}
```

## Implemented Optimization Passes

### 1. Constant Propagation (SCCP)

**File**: `compiler/src/mir/optimization/constant_propagation.rs`  
**Algorithm**: Sparse Conditional Constant Propagation (Wegman & Zadeck, 1991)

#### Overview

Replaces variables with known constant values at compile time, enabling further optimizations like dead code elimination and strength reduction.

#### Implementation Details

- **Lattice Analysis**: Uses 3-level lattice (Top → Constant → Bottom)
- **Worklist Algorithm**: Efficient iterative data flow analysis
- **Edge Profiling**: Tracks executable control flow edges
- **Conditional Handling**: Evaluates branches with known conditions

#### Key Methods

```rust
// Run SCCP analysis on a function
fn analyze(&mut self, func: &MirFunction)

// Evaluate instruction and update lattice values
fn evaluate_instruction(&mut self, func: &MirFunction, block: &MirBlock, instr: &MirInstruction)

// Replace instruction with constant if result is known
fn replace_with_constant(analysis: &SCCPAnalysis, instr: &MirInstruction) -> Option<MirInstruction>
```

#### Optimization Opportunities

- Constant folding: `x = 10 + 32` → `x = 42`
- Branch simplification: `if true { A } else { B }` → `A`
- Dead code detection: Code after unreachable branches

#### Configuration

```rust
pub struct ConstantPropagation;

impl MIRPass for ConstantPropagation {
    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String> {
        // Implements SCCP algorithm
    }
}
```

### 2. Dead Code Elimination (DCE)

**File**: `compiler/src/mir/optimization/dead_code_elimination.rs`  
**Algorithm**: Liveness Analysis + Data Flow Analysis

#### Overview

Removes code that does not affect program results, including dead assignments and unreachable blocks.

#### Implementation Details

- **Liveness Analysis**: Computes which values are live at each program point
- **Backward Data Flow**: Analyzes from program exits backwards
- **Two-Phase Elimination**: Removes dead assignments then unreachable blocks
- **SSA Preservation**: Maintains SSA form throughout

#### Key Methods

```rust
// Build use-def and def-use chains
fn build_use_def_maps(&mut self, func: &MirFunction) -> Result<(), String>

// Iterative liveness computation
fn compute_liveness(&mut self, func: &MirFunction) -> Result<(), String>

// Find dead code using liveness information
fn find_dead_code(func: &MirFunction, analysis: &LivenessAnalysis) -> (Vec<(usize, usize)>, HashSet<BlockId>)
```

#### Optimization Opportunities

- Dead assignments: `x = 10; return 42` → `return 42`
- Unreachable blocks: Code after `return` or infinite loops
- Unused phi nodes: Phi nodes whose results are unused

#### Performance Considerations

- O(n²) in worst case for large functions
- Worklist algorithm ensures reasonable performance in practice
- Memory usage scales with function complexity

### 3. Common Subexpression Elimination (CSE)

**File**: `compiler/src/mir/optimization/common_subexpression_elimination.rs`  
**Algorithm**: Available Expressions Analysis

#### Overview

Eliminates redundant computations by recognizing when expressions have been computed previously and can be reused.

#### Implementation Details

- **Available Expressions**: Computes set of expressions available at each point
- **Expression Hashing**: Normalizes expressions for comparison
- **Forward Analysis**: Propagates availability information forward
- **SSA-Aware**: Leverages SSA form for precise analysis

#### Key Methods

```rust
// Analyze available expressions in function
fn analyze_available_expressions(&mut self, func: &MirFunction)

// Check if expression is available at current point
fn is_expression_available(&self, expr: &Expression, block_id: BlockId) -> bool

// Replace redundant expression with saved result
fn replace_redundant_expression(&self, expr: &Expression, available_result: ValueId) -> Option<MirInstruction>
```

#### Expression Normalization

```rust
pub enum Expression {
    Binary { op: MirBinaryOp, left: ValueId, right: ValueId, ty: MirType },
    Load { address: ValueId, ty: MirType },
    // ... other expression types
}
```

#### Optimization Opportunities

- Arithmetic: `x = a + b; y = a + b` → `x = a + b; y = x`
- Memory loads: `y = *ptr; z = *ptr` → `y = *ptr; z = y`
- Function calls: Multiple calls with same arguments

### 4. Strength Reduction

**File**: `compiler/src/mir/optimization/strength_reduction.rs`  
**Algorithm**: Pattern-Based Replacement (Bodik & Wegman, 2000)

#### Overview

Replaces expensive operations with cheaper equivalents, particularly effective in loops and array access patterns.

#### Implementation Details

- **Induction Variable Detection**: Finds variables with regular update patterns
- **Loop Analysis**: Requires loop detection and analysis
- **Pattern Matching**: Recognizes expensive operation patterns
- **Replacement Strategy**: Substitutes with equivalent cheaper operations

#### Optimization Patterns

##### Division by Constant

```rust
// Replace: x / 8
// With: x >> 3  (for powers of 2)
x % 8  →  x & 7  (for powers of 2)
```

##### Multiplication Patterns

```rust
// Replace: x * 2
// With: x << 1
x * 4  →  x << 2
x * 8  →  x << 3
```

##### Array Indexing

```rust
// Replace: array[i * element_size]
// With: pointer arithmetic with increment
*base + i * 8  →  *(base + i) with base incrementing by 8
```

#### Key Methods

```rust
// Detect induction variables in loops
fn detect_induction_variable(&self, instr: &MirInstruction, natural_loop: &NaturalLoop) -> Option<InductionVariable>

// Replace expensive operations
fn replace_div_with_mul(&self, instr: &MirInstruction) -> Option<MirInstruction>
fn replace_mod_with_and(&self, instr: &MirInstruction) -> Option<MirInstruction>
```

### 5. Function Inlining

**File**: `compiler/src/mir/optimization/function_inlining.rs`  
**Algorithm**: Cost-Benefit Analysis (Briggs et al., 1998)

#### Overview

Replaces function calls with the body of the called function, eliminating call overhead and enabling further optimizations.

#### Implementation Details

- **Cost Model**: Evaluates benefit vs. cost of inlining
- **Call Graph Analysis**: Prevents infinite recursion
- **Size-Based Filtering**: Limits inlining to small functions
- **Heuristic Selection**: Chooses best candidates first

#### Cost Model

```rust
struct InliningCostModel {
    max_function_size: usize,      // Max instructions to inline
    max_depth: usize,             // Max call depth
    small_call_cost: f64,          // Cost for small functions
    medium_call_cost: f64,         // Cost for medium functions
    large_call_cost: f64,          // Cost for large functions
}
```

#### Selection Criteria

- Function size: Smaller functions preferred
- Call frequency: Hot calls preferred
- Call depth: Avoid deep recursion
- Return value complexity: Simple returns preferred

#### Key Methods

```rust
// Find inlining candidates
fn find_inline_candidates(&self, module: &MirModule) -> Vec<InlineCandidate>

// Estimate cost of inlining
fn estimate_cost(&self, func: &MirFunction, call_instr: &MirInstruction) -> f64

// Replace call with inline code
fn inline_call(&self, caller: &mut MirFunction, candidate: &InlineCandidate) -> Result<bool, String>
```

### 6. Epistemic Optimization

**File**: `compiler/src/mir/optimization/epistemic_optimization.rs`  
**Algorithm**: Sounio-Specific Optimization

#### Overview

Specialized optimizations for Sounio's epistemic types (Knowledge, Uncertain), preserving semantic invariants while improving performance.

#### Epistemic Types

```rust
enum EpistemicType {
    Knowledge,     // Knowledge<T> - certain knowledge
    Uncertain,     // Uncertain<T> - uncertain knowledge  
    Confidence,    // Confidence level (0.0-1.0)
    Uncertainty,   // Uncertainty measure
}
```

#### Invariant Preservation

- **Confidence Monotonicity**: Confidence only decreases through pure transforms
- **Uncertainty Propagation**: Uncertainty only increases without evidence fusion
- **Impossibility Detection**: Identifies epistemically impossible states

#### Optimization Opportunities

- **Certain Knowledge**: Optimize operations on 100% confidence values
- **Impossible States**: Eliminate code for impossible epistemic conditions
- **Confidence Propagation**: Maintain confidence through pure operations

#### Key Methods

```rust
// Analyze epistemic properties of values
fn analyze_epistemic_values(&mut self, func: &MirFunction)

// Optimize based on epistemic analysis
fn optimize_instruction(&self, instr: &MirInstruction) -> MirInstruction

// Detect impossible epistemic states
fn is_instruction_impossible(&self, instr: &MirInstruction) -> bool
```

### 7. SSA Validator

**File**: `compiler/src/mir/analysis/ssa_validator.rs`  
**Algorithm**: SSA Form Validation (Cytron et al., 1991)

#### Overview

Validates that MIR maintains proper SSA form properties, essential for correctness of SSA-based optimizations.

#### Validation Properties

##### 1. Single Assignment Property

Each variable is defined exactly once in the program.

##### 2. Dominance Property  

All uses of a variable are dominated by its definition.

##### 3. Phi Node Placement

Phi nodes are correctly placed at join points.

##### 4. Reachability

All code is reachable from program entry.

#### Implementation Details

- **Dominance Analysis**: Computes dominance relationships between blocks
- **Def-Use Chains**: Tracks definitions and uses of all values
- **CFG Analysis**: Analyzes control flow graph structure
- **Comprehensive Reporting**: Detailed error and warning messages

#### Key Methods

```rust
// Validate complete function for SSA properties
fn validate_function(&mut self, func: &MirFunction) -> SSAValidationResult

// Check single assignment property
fn check_single_assignment_property(&self, func: &MirFunction, errors: &mut Vec<SSAValidationError>)

// Verify dominance relationships
fn check_dominance_property(&self, func: &MirFunction, errors: &mut Vec<SSAValidationError>)

// Validate phi node placement
fn check_phi_node_placement(&self, func: &MirFunction, errors: &mut Vec<SSAValidationError>)
```

#### Validation Results

```rust
pub struct SSAValidationResult {
    pub is_valid: bool,
    pub errors: Vec<SSAValidationError>,
    pub warnings: Vec<SSAValidationWarning>,
}
```

## Pipeline Configuration

### Optimization Levels

```rust
pub enum OptimizationLevel {
    O0, // No optimization
    O1, // Basic optimization (Constant Propagation)
    O2, // Standard optimization (O1 + DCE + CSE)
    O3, // Aggressive optimization (O2 + Strength Reduction + Inlining)
}
```

### Pass Manager

```rust
pub struct PassManager {
    level: OptimizationLevel,
}

impl PassManager {
    pub fn run_function_passes(&mut self, func: &mut MirFunction) -> Result<PassResult, String> {
        // Applies passes based on optimization level
    }
}
```

### Default Pipeline

```rust
// O0: No optimization
// O1: Constant Propagation
// O2: + Dead Code Elimination + Common Subexpression Elimination  
// O3: + Strength Reduction + Function Inlining
```

## Usage Examples

### Basic Optimization

```rust
use sounio_compiler::mir::optimization::*;

// Create module
let mut module = build_mir_module();

// Run optimization
let mut pass_manager = PassManager::new_with_level(OptimizationLevel::O2);
let result = pass_manager.run_function_passes(&mut module)?;
```

### Custom Pass Sequence

```rust
let mut cp = ConstantPropagation::new();
let mut dce = DeadCodeElimination::new();
let mut cse = CommonSubexpressionElimination::new();

// Run passes in sequence
cp.run_on_module(&mut module)?;
dce.run_on_module(&mut module)?;
cse.run_on_module(&mut module)?;
```

### SSA Validation

```rust
use sounio_compiler::mir::analysis::ssa_validator::SSAValidator;

let mut validator = SSAValidator::new();
let result = validator.validate_function(&function);

if !result.is_valid {
    for error in result.errors {
        println!("SSA Error: {}", error.message);
    }
}
```

### Performance Benchmarking

```rust
use sounio_compiler::mir::benchmark::*;

let mut benchmarker = MIROptimizationBenchmarker::new(BenchmarkConfig::default());
let result = benchmarker.benchmark_pass("constant-propagation", module)?;

println!("Instructions: {} → {}", 
          result.instructions_before, 
          result.instructions_after);
```

## Development Guide

### Adding New Optimization Passes

1. **Implement MIRPass Trait**

```rust
pub struct MyOptimization {
    // Pass state
}

impl MIRPass for MyOptimization {
    fn name(&self) -> &'static str {
        "my-optimization"
    }
    
    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String> {
        // Implementation
    }
}
```

1. **Register in PassManager**

```rust
// In compiler/src/mir/optimization/mod.rs
match self.level {
    OptimizationLevel::O3 => {
        // Add to O3 pipeline
    }
    _ => {}
}
```

1. **Add Tests**

```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_my_optimization() {
        // Test implementation
    }
}
```

### Testing Strategy

#### Unit Tests

- Test individual optimization logic
- Validate edge cases and error handling
- Verify SSA preservation

#### Integration Tests  

- Test passes working together
- Validate end-to-end optimization pipeline
- Performance regression testing

#### Property-Based Tests

- Invariant preservation (SSA form, type safety)
- Optimization effectiveness
- Semantic preservation

### Performance Considerations

#### Memory Usage

- SSA form requires memory proportional to function complexity
- Large functions may need analysis throttling
- Pass-specific memory footprints vary significantly

#### Execution Time

- O(n²) worst case for some analyses (liveness, CSE)
- Worklist algorithms provide good practical performance
- Incremental analysis can improve performance

#### Optimization Quality

- Balance between compile time and runtime performance
- Profile-guided optimization for production builds
- Architecture-specific optimizations

## Debugging Tools

### SSA Validation

```rust
// Validate after each optimization
let mut validator = SSAValidator::new();
let result = validator.validate_function(&function);

if !result.is_valid {
    // Debug SSA violations
}
```

### Debug Output

```rust
// Enable verbose optimization logging
eprintln!("Optimization: Applied {} changes", changes.len());
```

### Benchmarking

```rust
// Profile optimization performance
let start = Instant::now();
let result = pass.run_on_function(&mut func)?;
let duration = start.elapsed();
```

## Future Extensions

### Planned Optimizations

- **Loop Unrolling**: Reduce loop overhead
- **Loop Fusion**: Combine compatible loops  
- **Alias Analysis**: Optimize memory operations
- **Register Allocation**: SSA-aware allocation
- **Vectorization**: SIMD optimization

### Machine Learning Integration

- **Profile-Guided Optimization**: Use execution profiles
- **Predictive Inlining**: ML-based inlining decisions
- **Architecture-Specific Tuning**: Auto-tune for target platform

### Advanced Analysis

- **Interprocedural Analysis**: Cross-function optimization
- **Whole-Program Analysis**: Global optimization opportunities
- **Precise Alias Analysis**: Better memory optimization

## References

1. Cytron, R., et al. (1991). "Efficiently Computing Static Single Assignment Form"
2. Muchnick, S. (1997). "Advanced Compiler Design and Implementation"
3. Aho, A., et al. (2007). "Compilers: Principles, Techniques, and Tools"
4. Bodik, R., & Wegman, M. (2000). "Strength Reduction"
5. Briggs, P., et al. (1998). "Threshold-based Greedy Function Inlining"
6. Knoop, J., et al. (1994). "Lazy Code Motion"

---

*This documentation covers the MIR optimization pipeline as of Sounio v0.93.0. For the most current implementation details, refer to the source code in `compiler/src/mir/`.*
