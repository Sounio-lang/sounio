<!-- docs:meta
topic_id: repo.docs.compiler.debugging-guide
authority: repo_only
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.compiler.debugging-guide
-->

# Debugging Guide for Sounio Programs

> **⚠️ File paths updated 2026-07-11 (doc-reality audit).** This page was written against the retired Rust compiler tree (`crates/`, `compiler/src/*.rs`, `codegen/llvm/`); those files no longer exist — the compiler is self-hosted Sounio (Madaros v0.80.0). The design and concepts below remain accurate, but DWARF/debug-info emission now lives in `self-hosted/native/dwarf.sio` and `self-hosted/native/debug_info.sio`; the epistemic/knowledge runtime in `self-hosted/compiler/knowledge_runtime_guard*.sio` — not any `codegen/llvm/*.rs` or `backend/native/*.rs`. Do not look for the `.rs` paths below.


## 1. Introduction

Debugging scientific code presents unique challenges compared to general-purpose programming, as it often involves not just correctness but also the reliability and interpretability of results. In Sounio programs, epistemic values—such as confidence levels and provenance—require special inspection to ensure that uncertainties are properly tracked and propagated. Refinement constraints, which enforce properties like positivity or bounds at the type level, demand validation to catch violations early. Units add another layer of complexity, as mismatches can lead to physically meaningless computations if not detected. To address these aspects in native code, Sounio leverages DWARF debug information, enabling detailed inspection of variables, types, and execution flow.

## 2. Setting Up Debug Environment

Establishing a robust debug environment is the foundation for effective troubleshooting in Sounio. This involves configuring compilation to generate necessary debug artifacts and selecting appropriate tools for inspection.

### 2.1 Compilation Flags

To enable debugging, use the `souc --debug` flag during compilation, which generates DWARF debug information for the native backend. This contrasts with release builds, which prioritize performance over inspectability. Optimization levels play a critical role: higher levels like `-O2` or `-O3` can obscure debuggability by inlining functions or reordering code, so stick to `-O0` or `-Og` for debug sessions. Importantly, preserving symbol names ensures that function and variable names remain readable in the debugger, avoiding mangled identifiers that complicate navigation.

### 2.2 Debug Info Formats

For the native backend targeting ELF binaries, DWARF serves as the primary debug format, embedding rich metadata about the program's structure. LLVM debug metadata provides an intermediate layer during code generation, facilitating source-level debugging. In interactive scenarios like the REPL, source maps link executed expressions back to original input, while line number preservation maintains traceability across transformations.

### 2.3 Required Tools

A suite of tools is essential for comprehensive debugging. GDB, the GNU Debugger, offers powerful command-line interaction for stepping through code and inspecting state. LLDB, the LLVM Debugger, provides similar capabilities with a more modern interface, particularly useful for LLVM-generated code. For performance analysis on Linux, perf enables profiling to identify bottlenecks. Valgrind detects memory errors such as leaks and invalid accesses, while rr supports record-and-replay debugging for deterministic reversal of execution.

## 3. DWARF Debug Information

DWARF debug information is central to inspecting Sounio's compiled binaries, providing a standardized way to map machine code back to source-level constructs, including the specialized types unique to scientific computing.

### 3.1 What is DWARF?

DWARF is a standard debug format widely used in Unix-like systems to describe program structure and data. It organizes information into sections like `.debug_info` for high-level descriptions, `.debug_line` for source line mappings, and `.debug_frame` for stack unwinding during exceptions. Type information is encoded to represent complex structures, and variable location tracking allows debuggers to follow values across function calls and scopes.

### 3.2 Generating DWARF in Sounio

In Sounio's implementation, DWARF generation occurs via the `DebugInfoBuilder` in `codegen/llvm/debug.rs`. This module creates `DIType` descriptors for Sounio's custom types, such as quantities with units or epistemic wrappers. `DIScope` and lexical blocks define the nesting of variables, while `DILocation` attaches source file and line information to instructions, enabling accurate breakpoints.

### 3.3 DWARF for Epistemic Types

Epistemic types like `Knowledge<T>` are encoded as structs in DWARF, capturing fields for value, confidence, and provenance. Field offsets ensure proper layout for full representations, including confidence intervals. To enhance usability, custom pretty-printers for GDB display these types intuitively, and visualizations distinguish full layouts (with all metadata) from compact ones (value-only for performance).

## 4. GDB Basics for Sounio

GDB is a versatile tool for debugging Sounio executables, allowing precise control over execution and inspection. Mastering its basics enables quick identification of issues in epistemic, refinement, or unit handling.

### 4.1 Starting GDB

Compile your program with debug flags and launch GDB as follows:

```bash
souc --debug program.sio -o program
gdb ./program
```

This loads the binary with full DWARF symbols, ready for interaction.

### 4.2 Essential GDB Commands

Key commands facilitate navigation and inspection. Set a breakpoint at entry with `break main`, then `run` to start execution. Use `step` to advance line-by-line (entering functions) or `next` to step over calls. Resume with `continue`. Inspect values via `print variable`, view the call stack with `backtrace` or `bt`, list local variables with `info locals`, and arguments with `info args`. Exit with `quit`.

### 4.3 Breakpoints

Breakpoints can target specific lines or functions. For source-level precision, use `break program.sio:42`. Function breakpoints like `break calculate_dose` halt at entry. Conditional variants, such as `break 42 if confidence < 0.9`, trigger only under specified criteria. Watchpoints monitor changes, e.g., `watch confidence` breaks on writes. Catchpoints handle exceptions with `catch throw`.

## 5. Inspecting Sounio Variables

Sounio's type system introduces specialized variables, from primitives to complex epistemic and unit-aware structures. GDB's `print` command reveals their internals, aiding in verification of computations.

### 5.1 Primitive Types

Basic scalars display straightforwardly:

```gdb
(gdb) print dose
$1 = 500.0
```

This confirms expected numerical values without additional metadata.

### 5.2 Knowledge Types (Full Layout)

Epistemic values in full layout expose all components:

```gdb
(gdb) print knowledge_var
$2 = {value = 36.5, confidence = 0.95, conf_lower = 36.3, conf_upper = 36.7,
      provenance_id = 12345, timestamp = 1706400000, revisable = 1}

(gdb) print knowledge_var.confidence
$3 = 0.95

(gdb) print knowledge_var.conf_upper - knowledge_var.conf_lower
$4 = 0.4
```

Accessing subfields or deriving metrics like interval width helps assess uncertainty propagation.

### 5.3 Quantity Types (Units)

Unit-aware types include dimensional annotations:

```gdb
(gdb) print concentration
$5 = {value = 25.0, unit = "mg/L", dimension = {M=1, L=-3, T=0, I=0, Θ=0, N=0, J=0}}
```

This reveals not just the magnitude but the physical dimensions, crucial for dimensional analysis.

### 5.4 Arrays and Slices

Collections require pointer-aware inspection:

```gdb
(gdb) print array[0]
(gdb) print *array@10    # Print first 10 elements
(gdb) print slice.len
(gdb) print slice.ptr
```

These commands display elements, lengths, and raw pointers, useful for bounds checking.

## 6. Debugging Epistemic Code

Epistemic features track uncertainty, but degradation or invalid states demand targeted debugging to maintain scientific integrity.

### 6.1 Confidence Assertions

Runtime checks enforce thresholds, as in:

```sio
assert_confidence(result, min: 0.95)  // Runtime check
```

Set a breakpoint on the assertion function:

```gdb
(gdb) break sounio_assert_confidence
(gdb) condition 1 actual_conf < 0.95
```

This halts only on failures, allowing examination of the offending value.

### 6.2 Tracing Confidence Degradation

Monitor changes dynamically:

```gdb
(gdb) watch result.confidence
(gdb) commands
> backtrace
> print result.confidence
> continue
> end
```

This script prints context each time confidence updates, revealing propagation paths.

### 6.3 Provenance Inspection

Trace origins with:

```gdb
(gdb) print result.provenance_id
(gdb) call sounio_provenance_trace(result.provenance_id)
# Shows full transformation chain
```

The trace reconstructs the computation history, identifying uncertainty sources.

### 6.4 Thermal Degradation Tracking

Environmental effects like heat impact confidence; inspect state with:

```gdb
(gdb) print thermal_state.current_temp_k
(gdb) print thermal_state.accumulated_cycles
(gdb) print thermal_state.accumulated_degradation
```

These fields quantify how runtime conditions erode reliability.

## 7. Debugging Refinement Types

Refinements embed constraints in types, catching errors at compile or runtime. Debugging focuses on violation detection and symbolic validation.

### 7.1 Constraint Violation Detection

A failing assignment triggers checks, e.g.:

```sio
type Positive = { x: i32 | x > 0 }
let bad: Positive = -5  // Runtime error
```

In GDB:

```gdb
(gdb) break sounio_refinement_check_failed
(gdb) run
# Breakpoint hit
(gdb) print value
$1 = -5
(gdb) print constraint
$2 = "x > 0"
```

This pinpoints the invalid value and its rule.

### 7.2 SMT Solver Integration

Validate symbolically:

```gdb
(gdb) call sounio_z3_check_sat(constraint)
# Returns SAT/UNSAT/UNKNOWN
```

Z3 integration confirms satisfiability, aiding complex constraint debugging.

### 7.3 Viewing Symbolic Constraints

Examine internal representations:

```gdb
(gdb) print refined_value.constraint_ast
# Shows Z3 AST representation
```

The AST visualizes the logical structure for manual verification.

## 8. Debugging Unit Errors

Units prevent dimensional inconsistencies, but mismatches require tracing to resolve.

### 8.1 Dimension Mismatch Detection

Incompatible operations fail, as in:

```sio
let mass: kg = 70.0
let length: m = 1.75
let wrong = mass + length  // ERROR
```

Debug with:

```gdb
(gdb) break sounio_unit_mismatch
(gdb) run
(gdb) print lhs_dimension
$1 = {M=1, L=0, T=0, I=0, Θ=0, N=0, J=0}  # kg
(gdb) print rhs_dimension
$2 = {M=0, L=1, T=0, I=0, Θ=0, N=0, J=0}  # m
```

Comparing dimensions highlights the incompatibility.

### 8.2 Conversion Tracing

Follow transformations:

```gdb
(gdb) break sounio_convert
(gdb) commands
> print from_unit
> print to_unit
> print conversion_factor
> continue
> end
```

This logs each conversion, verifying factors and units.

### 8.3 Runtime Dimension Validation

Check ad-hoc with:

```gdb
(gdb) call sounio_check_dimension(value, expected_dimension)
```

This function returns success or details on mismatches.

## 9. Advanced Debugging Techniques

Beyond basics, advanced methods like reversal and scripting unlock deeper analysis, especially for non-deterministic epistemic behaviors.

### 9.1 Reverse Debugging with rr

Record and replay execution for backward stepping:

```bash
rr record ./program
rr replay
(rr) break main
(rr) continue
(rr) reverse-step    # Step backwards!
(rr) reverse-continue
```

This determinism aids in isolating elusive bugs.

### 9.2 Time-Travel Debugging for Epistemic

In GDB, enable recording:

```gdb
(gdb) record
(gdb) continue
# Confidence degraded unexpectedly
(gdb) reverse-continue
(gdb) watch -l confidence  # Location watch
```

Reverse to the exact degradation point.

### 9.3 Conditional Breakpoints on Confidence

Target specific thresholds:

```gdb
(gdb) break calculate_dose if confidence < 0.8
(gdb) break if thermal_state.current_temp_k > 350.0
```

These automate detection of critical states.

### 9.4 Python GDB Scripts

Enhance output with custom printers in `.gdbinit`:

```python
# .gdbinit
import gdb

class KnowledgePrinter:
    def __init__(self, val):
        self.val = val

    def to_string(self):
        value = self.val['value']
        conf = self.val['confidence']
        lower = self.val['conf_lower']
        upper = self.val['conf_upper']
        return f"Knowledge({value:.3f} ± {(upper-lower)/2:.3f}, conf={conf:.3f})"

gdb.pretty_printers.append(KnowledgePrinter)
```

This formats epistemic types for readability.

## 10. Memory Debugging

Scientific codes handle large datasets, making memory issues a common pitfall. Specialized tools detect leaks and corruptions.

### 10.1 Valgrind for Memory Errors

Run with detailed checks:

```bash
valgrind --leak-check=full --track-origins=yes ./program
```

It reports leaks, invalid reads/writes, and origins.

### 10.2 AddressSanitizer (ASan)

Compile and execute with sanitization:

```bash
souc --debug --sanitize=address program.sio
./program
# Detects: use-after-free, buffer overflow, leaks
```

ASan instruments code for runtime detection without Valgrind's overhead.

### 10.3 Memory Layout Inspection

Examine raw bytes:

```gdb
(gdb) x/40xb &knowledge_var   # Examine 40 bytes in hex
(gdb) print sizeof(knowledge_var)
(gdb) print &knowledge_var
```

This verifies alignments and sizes for epistemic structs.

## 11. Performance Debugging

While debug builds sacrifice speed, profiling ensures optimizations don't introduce errors in epistemic or thermal modeling.

### 11.1 Profiling with perf

Balance debug and performance:

```bash
souc --debug --release program.sio -o program
perf record -g ./program
perf report
# Shows hotspots
```

The report highlights call graphs and hotspots.

### 11.2 Cycle Estimation Validation

Inspect backend analysis:

```gdb
(gdb) break native_backend::analyze_block
(gdb) print cycles.cycles
(gdb) print power.average_power_uw
```

This validates hardware modeling accuracy.

### 11.3 Thermal Profiling

Track heat effects:

```gdb
(gdb) print thermal_state.current_temp_k
(gdb) print thermal_state.accumulated_cycles
(gdb) call thermal_state.history()
```

History calls log temporal profiles.

## 12. REPL Debugging

The Sounio REPL supports interactive debugging, ideal for quick experiments with refinements and epistemic ops.

### 12.1 REPL Error Inspection

Errors provide inline diagnostics:

```sio
sounio> let dose: mg = -5.0  // Invalid
Error: Refinement constraint violated
  type Positive = { x: f64 | x > 0 }
  got: -5.0

sounio> :debug on
sounio> let dose: mg = -5.0
[Debug] Entering refinement check
[Debug] Value: -5.0
[Debug] Constraint: x > 0
[Debug] Result: FAILED
```

Debug mode adds step-by-step traces.

### 12.2 Interactive Inspection

Query structures directly:

```sio
sounio> let k: Knowledge<f64> = measure(5.0, 0.1)
sounio> :inspect k
Knowledge<f64> {
  value: 5.0,
  confidence: 0.95,
  interval: [4.9, 5.1],
  provenance: Measurement(sensor_id=42, timestamp=...)
}
```

This formats output for clarity.

### 12.3 Tracing Execution

Enable traces for flows:

```sio
sounio> :trace on
sounio> let result = calculate_dose(weight, creatinine)
[Trace] calculate_dose(70.0 kg, 45.0 mL/min)
[Trace]   -> clearance: 45.0 mL/min
[Trace]   -> dose_factor: 0.375
[Trace]   -> result: 187.5 mg (conf=0.92)
```

Traces log key computations.

## 13. Common Debugging Scenarios

Recurring issues in Sounio often stem from epistemic drift, unit errors, or environmental factors. Targeted strategies resolve them efficiently.

### 13.1 "Confidence Too Low" Errors

Halt on assertions:

```gdb
(gdb) break assert_confidence
(gdb) backtrace
# Trace back to see which operations degraded confidence
(gdb) print result.confidence
(gdb) call sounio_provenance_trace(result.provenance_id)
```

Backtrace and provenance reveal degradation sources.

### 13.2 Unit Mismatch in PK/PD

Catch pharmacometric errors:

```gdb
(gdb) break sounio_unit_error
(gdb) print expected.dimension
(gdb) print actual.dimension
(gdb) call suggest_conversion(actual, expected)
```

Suggestions guide fixes.

### 13.3 Thermal Runaway

Monitor thresholds:

```gdb
(gdb) watch thermal_state.current_temp_k
(gdb) condition 1 thermal_state.current_temp_k > 350.0
(gdb) run
# Breaks when temperature exceeds threshold
(gdb) backtrace
```

This detects overheating early.

### 13.4 Refinement Assertion Failures

Probe solver calls:

```gdb
(gdb) break Z3_check_sat
(gdb) run
(gdb) print constraint_expr
(gdb) call Z3_ast_to_string(context, constraint_expr)
```

String conversion aids constraint debugging.

## 14. Debugging GPU Kernels

For GPU-accelerated Sounio code, CUDA-specific tools extend debugging to parallel execution.

### 14.1 CUDA-GDB for PTX

Compile and debug:

```bash
souc --gpu=cuda --debug kernel.sio
cuda-gdb ./program
(cuda-gdb) break kernel_function
(cuda-gdb) set cuda api_failures stop
```

This stops on API errors.

### 14.2 Inspecting Thread State

Query parallelism:

```gdb
(cuda-gdb) info cuda threads
(cuda-gdb) cuda thread 0,0,0  # Block 0, Thread 0
(cuda-gdb) print thread_id.x
```

Thread selection isolates issues.

### 14.3 Memory Transfer Validation

Monitor transfers:

```gdb
(cuda-gdb) break cudaMemcpy
(cuda-gdb) print src
(cuda-gdb) print dst
(cuda-gdb) print count
```

This verifies data movement.

## 15. Logging and Tracing

Logs complement debuggers by capturing non-interactive details like compiler passes or runtime events.

### 15.1 Compiler Logging

Enable verbose output:

```bash
SOUNIO_LOG=debug souc program.sio
# Shows: type inference, effect resolution, optimization passes
```

This traces frontend and backend steps.

### 15.2 Runtime Logging

Capture execution:

```bash
SOUNIO_RUNTIME_LOG=trace ./program
# Shows: epistemic operations, unit conversions, refinement checks
```

Traces highlight dynamic behaviors.

### 15.3 Provenance Logging

Log full histories:

```bash
SOUNIO_PROVENANCE=full ./program
# Full transformation DAG logged to file
```

The DAG file enables offline analysis.

## 16. Debugging Best Practices

Adopting structured practices ensures reproducible and efficient debugging across Sounio's scientific workflows.

### 16.1 Reproducibility

Maintain determinism by using fixed random seeds, recording inputs for replay, versioning test cases in control systems, and documenting the environment, including compiler versions and dependencies.

### 16.2 Assertion Placement

Strategically place checks: preconditions validate inputs, postconditions verify outputs, invariants guard loops, and confidence thresholds monitor epistemic health.

### 16.3 Debug Builds vs Release

Debug builds include full symbols, disable optimizations, and enable assertions for thorough inspection. Release builds apply optimizations but retain DWARF via `--debug`; test both to catch mode-specific issues.

## 17. IDE Integration

IDEs streamline debugging by integrating tools with editors, offering visual aids for Sounio code.

### 17.1 VSCode + CodeLLDB

Configure launches for breakpoints, variable inspection in side panels, inline value display, and a debug console for expressions.

### 17.2 IntelliJ/CLion

Leverage the native debugger for memory views, expression evaluation, and call stack navigation, ideal for large projects.

### 17.3 Vim/Emacs + GDB/LLDB

Integrate via terminals with GDB dashboard for enhanced layouts and source window synchronization for seamless editing.

## 18. Error Message Interpretation

Sounio's errors are informative, guiding users from symptoms to fixes in types, refinements, and epistemics.

### 18.1 Type Errors

Mismatches include hints:

```
Error: Type mismatch in assignment
  Expected: Knowledge<mg/L>
  Got:      Knowledge<mg/mL>

  Hint: Did you mean to convert units?
  let conc: mg/L = convert(value, to: mg/L)
```

Suggestions point to conversions.

### 18.2 Refinement Errors

Violations specify details:

```
Error: Refinement constraint violated
  Type: Positive
  Constraint: x > 0
  Value: -5.0

  at program.sio:42:15
  let dose: Positive = calculate(-5.0)
                       ^
```

Line pointers aid location.

### 18.3 Epistemic Errors

Threshold breaches trace chains:

```
Error: Confidence below required threshold
  Required: 0.95
  Actual:   0.72

  Provenance trace:
    1. Measurement(sensor=A, conf=0.98)
    2. Multiply(operand=B, conf=0.85) -> conf=0.91
    3. Add(operand=C, conf=0.80) -> conf=0.85
    4. ThermalDegradation(cycles=1M, temp=350K) -> conf=0.72
```

Traces explain degradation.

## 19. Testing Infrastructure

Integrate debugging into tests to catch regressions in epistemic propagation, units, and refinements.

### 19.1 Unit Tests with Debug Info

Embed inspections:

```rust
#[test]
fn test_epistemic_propagation_debug() {
    let a = KnowledgeFull::new(5.0, 0.95, 4.9, 5.1, 1, 0, false);
    let b = KnowledgeFull::new(3.0, 0.90, 2.9, 3.1, 2, 0, false);

    let mut result = KnowledgeFull::constant(0.0);
    unsafe {
        sounio_epistemic_add_full(&a, &b, &mut result);
    }

    assert!((result.confidence - 0.9236).abs() < 0.001);  // sqrt(0.95*0.90)

    // Debug inspection
    eprintln!("Result: {:?}", result);
}
```

Prints aid manual verification.

### 19.2 Integration Tests

Run with output:

```bash
cargo test --test integration_debug -- --nocapture
```

Uncaptured output shows debug info.

### 19.3 Regression Tests

Capture failures, add to suites, and verify fixes via debugger runs to prevent recurrence.

## 20. Resources

### 20.1 Documentation

- GDB manual: https://sourceware.org/gdb/documentation/
- DWARF standard: http://dwarfstd.org/
- LLDB tutorial: https://lldb.llvm.org/use/tutorial.html

### 20.2 Tools

- GDB dashboard: https://github.com/cyrus-and/gdb-dashboard
- rr: https://rr-project.org/
- Valgrind: https://valgrind.org/

### 20.3 Sounio-Specific

- compiler/src/codegen/llvm/debug.rs - DWARF generation
- compiler/src/backend/native/epistemic_runtime.rs - Runtime functions
- compiler/tests/ - Example debug scenarios
