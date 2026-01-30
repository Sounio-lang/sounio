# Effect Backend Integration Implementation Plan

**Status**: In Progress
**Created**: 2026-01-30
**Phases**: B → A → D → C

---

## Overview

Integrate the fully-implemented effect handler system with compiled backends (Native, LLVM, Cranelift) to enable zero-cost effect handling in production code.

**Current State**:
- ✅ Interpreter: Fully working with 13 effect handlers (14,263 LOC)
- ⚠️ Cranelift/LLVM: Partial - dispatch works, continuations don't
- 🚧 Native: Stubbed - foundation exists, no execution

**Goal**: Production-ready compiled effect handlers with continuation support across all backends.

---

## PHASE B: Backend Integration (Current)

**Objective**: Wire up existing effect infrastructure to compiled backends

### B.1: Native Backend Foundation ⚡ PRIORITY

**File**: `/crates/souc/src/backend/native/effects.rs`

#### B.1.1: Implement External Function Linking
- [ ] Complete `bl_external()` to emit real BL/CALL instructions
- [ ] Add relocation table entries for external symbols
- [ ] Implement symbol resolution in assembler
- [ ] Add runtime function address injection

**Current Code**:
```rust
fn bl_external(&mut self, func_name: &str, args: &[Reg]) {
    // TODO: emit real BL instruction with relocation
    self.code.push(0xD503201F); // NOP placeholder
}
```

**Target**:
```rust
fn bl_external(&mut self, func_name: &str, args: &[Reg]) {
    let symbol_id = self.add_external_symbol(func_name);
    let offset = self.create_relocation(symbol_id, RelocationType::BranchLink);
    self.emit_bl_with_reloc(offset); // BL instruction with R_AARCH64_CALL26
}
```

#### B.1.2: Runtime Function Compilation
- [ ] Create `/crates/runtime/Cargo.toml` build script
- [ ] Compile runtime functions to object files
- [ ] Generate `.a` static library with all symbols
- [ ] Add linker flags to include runtime library

**Build Flow**:
```
runtime/src/handler_stack.rs (Rust)
    ↓ rustc --emit=obj
runtime.o
    ↓ ar rcs
libsounio_runtime.a
    ↓ link with compiled program
final executable with __sounio_dispatch_* symbols
```

#### B.1.3: Continuation Capture (Basic)
- [ ] Implement `NativeContinuation::capture()` for AArch64
- [ ] Save general-purpose registers (X0-X30)
- [ ] Save floating-point registers (D0-D31)
- [ ] Save stack pointer and frame pointer
- [ ] Snapshot stack frames (caller's activation record)

**Assembly Stub** (AArch64):
```rust
pub unsafe fn capture() -> NativeContinuation {
    let mut cont = NativeContinuation::new();
    asm!(
        "stp x0, x1, [{cont_ptr}, #0]",   // Save X0-X1
        "stp x2, x3, [{cont_ptr}, #16]",  // Save X2-X3
        // ... continue for all GP registers
        "mov {sp}, sp",                    // Capture SP
        "mov {fp}, x29",                   // Capture FP
        cont_ptr = in(reg) &mut cont,
        sp = out(reg) cont.stack_pointer,
        fp = out(reg) cont.frame_pointer,
    );
    cont
}
```

#### B.1.4: Continuation Resumption (Basic)
- [ ] Implement `NativeContinuation::resume()` for AArch64
- [ ] Restore registers from continuation struct
- [ ] Restore stack pointer
- [ ] Jump to return address (BR instruction)
- [ ] Handle resume count tracking

**Assembly Stub**:
```rust
pub unsafe fn resume(&mut self, value: f64) -> ! {
    asm!(
        "ldp x0, x1, [{cont_ptr}, #0]",   // Restore X0-X1
        // ... restore all registers
        "mov sp, {sp}",                    // Restore SP
        "mov x0, {value}",                 // Pass resume value in X0
        "br {return_addr}",                // Jump to return address
        cont_ptr = in(reg) self,
        sp = in(reg) self.stack_pointer,
        value = in(reg) value,
        return_addr = in(reg) self.return_address,
        options(noreturn)
    );
}
```

### B.2: Cranelift Integration

**File**: `/crates/souc/src/codegen/cranelift.rs`

#### B.2.1: Enhance Effect Dispatch
- [ ] Add continuation capture before dispatch call
- [ ] Pass continuation ID to runtime functions
- [ ] Implement handler callback registration
- [ ] Add resume point tracking in JIT state

**Current**:
```rust
Op::PerformEffect { effect, op, args } => {
    let result = self.call_runtime_fn("dispatch_io_print", &[value]);
}
```

**Target**:
```rust
Op::PerformEffect { effect, op, args } => {
    let cont_id = self.capture_continuation();
    let result = self.call_runtime_fn_with_cont(
        "dispatch_io_print",
        &[value, cont_id]
    );
    self.register_resume_point(cont_id, current_block);
}
```

#### B.2.2: Multi-Shot Continuation Support
- [ ] Clone continuation for multi-shot effects
- [ ] Track resume count in `JitEffectState`
- [ ] Implement continuation store integration
- [ ] Add GC for unresumed continuations

### B.3: LLVM Integration

**File**: `/crates/souc/src/codegen/llvm/codegen.rs`

#### B.3.1: Generic Dispatch Enhancement
- [ ] Add continuation parameter to `runtime_dispatch_generic`
- [ ] Emit LLVM IR to capture execution context
- [ ] Implement resume callback mechanism
- [ ] Add exception handling for resume failures

**LLVM IR Target**:
```llvm
define double @perform_effect(i8* %effect, i8* %op, double* %args, i64 %len) {
entry:
    %cont_id = call i64 @__sounio_capture_continuation()
    %result = call double @__sounio_dispatch_generic(
        i8* %effect,
        i8* %op,
        double* %args,
        i64 %len,
        i64 %cont_id
    )
    ret double %result
}
```

#### B.3.2: LLVM Continuation Intrinsics
- [ ] Research LLVM coroutine intrinsics (`llvm.coro.*`)
- [ ] Evaluate if we can use built-in coroutine support
- [ ] Implement custom intrinsics if needed
- [ ] Add continuation cleanup passes

### B.4: Runtime Handler Stack Enhancement

**File**: `/crates/runtime/src/handler_stack.rs`

#### B.4.1: Continuation-Aware Dispatch
- [ ] Update all `__sounio_dispatch_*` signatures to accept continuation ID
- [ ] Store continuation in handler state
- [ ] Implement resume callback registration
- [ ] Add continuation lifecycle management

**Before**:
```rust
#[no_mangle]
pub extern "C" fn __sounio_dispatch_io_print(value: f64) -> f64 {
    // ... handler logic
}
```

**After**:
```rust
#[no_mangle]
pub extern "C" fn __sounio_dispatch_io_print(
    value: f64,
    cont_id: u64
) -> f64 {
    let cont = CONTINUATION_STORE.get(cont_id);
    // ... handler logic that can call cont.resume(result)
}
```

#### B.4.2: Handler State Persistence
- [ ] Add continuation ID to handler state
- [ ] Implement state snapshot/restore
- [ ] Add multi-handler nesting support
- [ ] Implement handler cleanup on panic

### B.5: Testing Infrastructure

#### B.5.1: Native Backend Tests
- [ ] Test external function linking
- [ ] Test continuation capture/resume
- [ ] Test register save/restore correctness
- [ ] Test stack integrity across suspend/resume

#### B.5.2: End-to-End Compiled Effect Tests
- [ ] Port interpreter effect tests to compiled mode
- [ ] Test one-shot effects (IO, print)
- [ ] Test multi-shot effects (Prob, Amb)
- [ ] Test nested handlers
- [ ] Test handler cleanup

#### B.5.3: Cross-Backend Compatibility
- [ ] Test same program on all three backends
- [ ] Verify result equivalence
- [ ] Benchmark performance differences

**Test Files**:
- `/tests/compiled/test_native_effects.rs`
- `/tests/compiled/test_cranelift_effects.rs`
- `/tests/compiled/test_llvm_effects.rs`
- `/tests/compiled/test_cross_backend.rs`

---

## PHASE A: CPS Transformation

**Objective**: Implement zero-cost effect handlers via CPS compilation

### A.1: Complete CPS Transform

**File**: `/crates/souc/src/backend/cps_transform.rs`

#### A.1.1: Terminator Transformation
- [ ] Complete `transform_terminator()` to emit tail calls
- [ ] Convert returns to continuation calls
- [ ] Transform branches to pass continuations
- [ ] Handle loops with continuation threading

**Current**:
```rust
fn transform_terminator(&mut self, term: &HlirTerminator) -> HlirTerminator {
    term.clone() // TODO: transform
}
```

**Target**:
```rust
fn transform_terminator(&mut self, term: &HlirTerminator) -> HlirTerminator {
    match term {
        HlirTerminator::Return(value) => {
            // return value => k(value)
            HlirTerminator::TailCall(cont_param, vec![value.clone()])
        }
        HlirTerminator::Branch { cond, then_bb, else_bb } => {
            // Transform branches to continuation-passing
            self.transform_branch(cond, then_bb, else_bb)
        }
        // ... other terminators
    }
}
```

#### A.1.2: Effect Operation Transformation
- [ ] Transform `perform` to explicit continuation capture
- [ ] Add continuation parameter to effectful calls
- [ ] Implement handler installation/removal in CPS
- [ ] Handle nested effect handlers

**Before**:
```rust
let result = perform IO.print("hello");
process(result)
```

**After (CPS)**:
```rust
perform IO.print("hello", k: |result| {
    process(result)
})
```

#### A.1.3: Function Signature Transformation
- [ ] Add continuation parameter to effectful functions
- [ ] Update call sites to pass continuations
- [ ] Handle higher-order functions
- [ ] Preserve non-effectful functions (no transform)

**Before**:
```sio
fn read_and_process() -> i32 with IO {
    let line = perform IO.read_line();
    parse(line)
}
```

**After (internal CPS IR)**:
```rust
fn read_and_process(k: Continuation<i32>) with IO {
    perform IO.read_line(|line| {
        let result = parse(line);
        k(result)
    })
}
```

### A.2: Integration with Codegen Pipeline

#### A.2.1: Add CPS Pass to Pipeline
- [ ] Call `CpsTransform::transform()` after HIR → SIR
- [ ] Add feature flag for CPS vs closure-based effects
- [ ] Implement selective CPS (only effectful functions)
- [ ] Add validation pass after CPS

**Pipeline**:
```
AST → HIR → SIR → HLIR (SSA)
                    ↓
              CPS Transform (if enabled)
                    ↓
              Codegen (Native/LLVM/Cranelift)
```

#### A.2.2: Effect Analysis
- [ ] Implement effect inference for functions
- [ ] Build effect dependency graph
- [ ] Mark CPS-transformed functions in metadata
- [ ] Optimize away CPS for pure functions

### A.3: Optimization Passes

#### A.3.1: Tail Call Optimization
- [ ] Recognize tail-continuation calls
- [ ] Emit direct jumps instead of calls
- [ ] Optimize continuation allocation
- [ ] Implement continuation inlining

#### A.3.2: Continuation Allocation Elimination
- [ ] Stack-allocate continuations where possible
- [ ] Eliminate allocations for one-shot continuations
- [ ] Implement region-based memory management
- [ ] Add escape analysis for continuations

#### A.3.3: Effect Fusion
- [ ] Fuse consecutive effect operations
- [ ] Batch handler dispatch calls
- [ ] Eliminate redundant handler push/pop
- [ ] Optimize nested handler sequences

### A.4: Testing CPS Transformation

#### A.4.1: Correctness Tests
- [ ] Test simple effectful functions
- [ ] Test nested effects
- [ ] Test higher-order functions with effects
- [ ] Test recursive functions with effects

#### A.4.2: Performance Tests
- [ ] Benchmark CPS vs interpreter
- [ ] Benchmark CPS vs closure-based
- [ ] Measure continuation allocation overhead
- [ ] Profile tail call optimization impact

---

## PHASE D: Production Hardening

**Objective**: Replace simulations with real integrations

### D.1: GPU Integration

**File**: `/crates/souc/src/effects/handlers/gpu.rs`

#### D.1.1: CUDA Backend
- [ ] Add `cuda-sys` dependency
- [ ] Implement `cuInit`, `cuDeviceGet`, `cuCtxCreate`
- [ ] Implement actual kernel launch via `cuLaunchKernel`
- [ ] Add device memory management (cuMemAlloc, cuMemcpy)
- [ ] Implement stream management

#### D.1.2: HIP Backend (AMD)
- [ ] Add `hip-sys` dependency
- [ ] Implement HIP runtime calls
- [ ] Add ROCm compatibility layer
- [ ] Test on AMD hardware

#### D.1.3: Metal Backend (Apple)
- [ ] Add `metal-rs` dependency
- [ ] Implement Metal command buffers
- [ ] Add shader compilation
- [ ] Test on macOS/iOS

#### D.1.4: Vulkan Compute
- [ ] Add `ash` (Vulkan bindings) dependency
- [ ] Implement compute pipeline creation
- [ ] Add buffer/descriptor management
- [ ] Cross-platform testing

### D.2: Network Integration

**File**: `/crates/souc/src/effects/handlers/network.rs`

#### D.2.1: HTTP Client
- [ ] Replace simulation with `reqwest`
- [ ] Implement actual GET/POST/PUT/DELETE
- [ ] Add timeout support
- [ ] Implement retry logic with exponential backoff

#### D.2.2: WebSocket Support
- [ ] Replace simulation with `tokio-tungstenite`
- [ ] Implement connect/send/receive/close
- [ ] Add ping/pong heartbeat
- [ ] Handle reconnection

#### D.2.3: DNS Resolution
- [ ] Implement actual DNS lookup
- [ ] Add caching layer
- [ ] Support custom resolvers
- [ ] Handle DNSSEC

### D.3: Async Runtime Integration

**File**: `/crates/souc/src/effects/handlers/async.rs`

#### D.3.1: Tokio Integration
- [ ] Replace simulation with actual `tokio::spawn`
- [ ] Implement real await mechanism
- [ ] Add task cancellation
- [ ] Integrate with tokio scheduler

#### D.3.2: Future Conversion
- [ ] Convert continuations to Rust Futures
- [ ] Implement `Future` trait for effect continuations
- [ ] Add async/await syntax sugar
- [ ] Support `async fn` in Sounio

#### D.3.3: Multi-Threaded Runtime
- [ ] Add thread pool configuration
- [ ] Implement work stealing
- [ ] Add task priority support
- [ ] Integrate with system scheduler

### D.4: Error Handling & Resilience

#### D.4.1: Graceful Degradation
- [ ] Add fallback mechanisms for failed operations
- [ ] Implement circuit breaker pattern
- [ ] Add health checks
- [ ] Implement timeout policies

#### D.4.2: Logging & Observability
- [ ] Add structured logging with `tracing`
- [ ] Implement distributed tracing
- [ ] Add metrics collection (Prometheus)
- [ ] Create dashboards for effect operations

#### D.4.3: Security Hardening
- [ ] Add TLS/SSL for network operations
- [ ] Implement authentication/authorization
- [ ] Add rate limiting
- [ ] Sanitize inputs

### D.5: Production Testing

#### D.5.1: Integration Tests
- [ ] Test real GPU kernel execution
- [ ] Test real HTTP requests (against test server)
- [ ] Test real WebSocket connections
- [ ] Test async tasks with tokio

#### D.5.2: Load Testing
- [ ] Benchmark GPU kernel launch overhead
- [ ] Test network handler under load
- [ ] Profile async task scheduling
- [ ] Measure memory usage under stress

#### D.5.3: Chaos Engineering
- [ ] Test network failures
- [ ] Test GPU out-of-memory scenarios
- [ ] Test async task cancellation
- [ ] Verify cleanup on panics

---

## PHASE C: Performance Optimization

**Objective**: Maximize performance of effect handlers

### C.1: Benchmarking Infrastructure

#### C.1.1: Microbenchmarks
- [ ] Create `benches/effect_dispatch.rs`
- [ ] Benchmark handler dispatch overhead
- [ ] Benchmark continuation capture/resume
- [ ] Benchmark one-shot vs multi-shot

#### C.1.2: Macro Benchmarks
- [ ] Benchmark real-world programs with effects
- [ ] Compare interpreter vs compiled backends
- [ ] Compare CPS vs closure-based
- [ ] Compare to native Rust (zero-cost baseline)

#### C.1.3: Profiling
- [ ] Add `perf` integration
- [ ] Create flame graphs for effect operations
- [ ] Identify hotspots
- [ ] Profile cache behavior

### C.2: Optimization Strategies

#### C.2.1: Handler Dispatch Optimization
- [ ] Implement inline effect dispatch
- [ ] Add effect specialization
- [ ] Implement handler devirtualization
- [ ] Add branch prediction hints

**Current** (indirect call):
```rust
let handler = registry.get(effect);
handler.handle(op, args);
```

**Optimized** (direct call):
```rust
// Compiler knows exact handler at compile time
__sounio_io_print_inline(value);
```

#### C.2.2: Continuation Optimization
- [ ] Stack-allocate continuations (escape analysis)
- [ ] Eliminate continuations for non-resuming handlers
- [ ] Implement continuation pooling
- [ ] Add LLVM's coroutine optimization passes

#### C.2.3: Effect Fusion & Batching
- [ ] Fuse consecutive effect operations
- [ ] Batch handler push/pop operations
- [ ] Optimize nested handlers
- [ ] Eliminate redundant handler lookups

**Before**:
```sio
perform IO.print("a");
perform IO.print("b");
perform IO.print("c");
```

**After** (fused):
```sio
perform IO.print_batch(["a", "b", "c"]);
```

#### C.2.4: Zero-Cost Abstractions
- [ ] Ensure monomorphization of effect handlers
- [ ] Inline small handler operations
- [ ] Eliminate runtime type checks
- [ ] Specialize for common effect patterns

### C.3: Backend-Specific Optimizations

#### C.3.1: Native Backend
- [ ] Implement register-passing convention for continuations
- [ ] Use callee-save registers for handler state
- [ ] Emit SIMD instructions for batch operations
- [ ] Add CPU-specific optimizations (ARM NEON, x86 AVX)

#### C.3.2: LLVM Backend
- [ ] Enable LLVM optimization passes (O2/O3)
- [ ] Add custom LLVM passes for effects
- [ ] Implement link-time optimization (LTO)
- [ ] Add profile-guided optimization (PGO)

#### C.3.3: Cranelift Backend
- [ ] Enable Cranelift optimization settings
- [ ] Implement effect-specific Cranelift passes
- [ ] Add JIT code caching
- [ ] Implement tiered compilation

### C.4: Memory Optimization

#### C.4.1: Continuation Memory Management
- [ ] Implement continuation pooling
- [ ] Add arena allocation for handlers
- [ ] Optimize stack snapshot size
- [ ] Implement copy-on-write for multi-shot

#### C.4.2: Handler State Optimization
- [ ] Use bump allocator for handler state
- [ ] Implement state compaction
- [ ] Add state sharing across handlers
- [ ] Optimize state serialization

### C.5: Performance Testing

#### C.5.1: Regression Testing
- [ ] Create performance baseline
- [ ] Add CI performance checks
- [ ] Track performance metrics over time
- [ ] Alert on regressions

#### C.5.2: Comparison Studies
- [ ] Compare to Koka (effect system language)
- [ ] Compare to OCaml with Affect
- [ ] Compare to Rust with async/await
- [ ] Compare to Julia (similar domain)

---

## Implementation Priority

### Week 1: Foundation (B.1)
- Native backend linking (B.1.1)
- Runtime compilation (B.1.2)
- Basic continuation capture (B.1.3)

### Week 2: Integration (B.2, B.3, B.4)
- Cranelift continuation support
- LLVM continuation support
- Runtime enhancement
- Initial tests

### Week 3-4: CPS Transformation (A.1, A.2)
- Complete CPS transform
- Integrate into pipeline
- Testing

### Week 5-6: Production Hardening (D.1, D.2, D.3)
- GPU integration
- Network integration
- Async integration

### Week 7-8: Optimization (C.1, C.2, C.3)
- Benchmarking
- Optimization passes
- Performance tuning

---

## Success Metrics

### Correctness
- [ ] All interpreter tests pass in compiled mode
- [ ] Multi-shot effects work correctly
- [ ] Handler cleanup is guaranteed
- [ ] No memory leaks in continuation management

### Performance
- [ ] CPS compiled code within 5% of hand-written Rust
- [ ] Handler dispatch overhead < 10ns (native backend)
- [ ] Continuation capture < 50ns (native backend)
- [ ] No performance regression vs interpreter for small programs

### Completeness
- [ ] All 13 effects work in all 3 backends
- [ ] Real integrations (GPU/Network/Async) functional
- [ ] Production-ready error handling
- [ ] Comprehensive documentation

---

## Dependencies

### External Crates
- `cuda-sys` (GPU/CUDA)
- `hip-sys` (GPU/ROCm)
- `metal-rs` (GPU/Metal)
- `ash` (Vulkan compute)
- `reqwest` (HTTP client)
- `tokio-tungstenite` (WebSocket)
- `tokio` (async runtime)
- `tracing` (logging)
- `criterion` (benchmarking)

### Internal Modules
- Effect handler registry (existing)
- Continuation store (existing)
- Handler capability trait (existing)
- CPS transform (partial)
- Native backend (partial)

---

## Documentation

### User Documentation
- [ ] Effect system guide with compiled examples
- [ ] Performance characteristics guide
- [ ] Backend selection guide
- [ ] Best practices for effect usage

### Developer Documentation
- [ ] Backend integration guide
- [ ] CPS transformation internals
- [ ] Continuation capture mechanism
- [ ] Adding new effect handlers

### Research Documentation
- [ ] Performance comparison study
- [ ] CPS vs closure-based analysis
- [ ] Linearity enforcement evaluation
- [ ] Novel contributions paper

---

## Risk Assessment

### High Risk
1. **Platform-specific assembly** - AArch64/x86-64 differences
   - Mitigation: Abstract register allocation, use Rust `asm!` macro
2. **Linking complexity** - Runtime symbol resolution
   - Mitigation: Use standard linker, create proper .a library
3. **Continuation correctness** - Stack corruption risks
   - Mitigation: Extensive testing, memory sanitizers

### Medium Risk
1. **Performance overhead** - CPS may be slower than expected
   - Mitigation: Benchmark early, optimize iteratively
2. **LLVM coroutine integration** - May not fit our model
   - Mitigation: Custom intrinsics as fallback
3. **GPU driver compatibility** - CUDA/HIP version issues
   - Mitigation: Target stable API versions, graceful fallback

### Low Risk
1. **Network integration** - Well-understood with `reqwest`
2. **Async integration** - Tokio is battle-tested
3. **Testing infrastructure** - Can reuse existing tests

---

## Notes

- **Phase B** is the critical path - everything depends on backend integration
- **Phase A** enables zero-cost abstractions but isn't strictly required
- **Phase D** is about production readiness, can be incremental
- **Phase C** is continuous improvement

Estimated total effort: **8-10 weeks** for full implementation and testing.
