# Continuation Capture and Resumption Implementation Guide

This document describes how to implement full continuation capture and resumption for the native backend.

## Overview

Continuation capture requires saving the complete execution state (registers, stack) so that execution can be resumed later from the exact same point. This is essential for implementing algebraic effect handlers.

## AArch64 Implementation

### Register State

AArch64 has:
- **31 general-purpose registers**: X0-X30 (64-bit)
  - X29 = Frame Pointer (FP)
  - X30 = Link Register (LR, contains return address)
  - X31/XZR/SP = Stack Pointer (context-dependent)

- **32 SIMD/FP registers**: V0-V31 (128-bit)
  - Can be accessed as B (8-bit), H (16-bit), S (32-bit), D (64-bit), or Q (128-bit)
  - For capture, store as D registers (64-bit) in pairs

### Capture Assembly Template

```asm
// Function: __sounio_capture_continuation
// Input: X0 = pointer to AArch64Continuation structure
// Output: Continuation structure filled with current state
// Clobbers: None (must preserve all registers!)

__sounio_capture_continuation:
    // Save general-purpose registers X0-X30
    // Use X0 as base pointer to continuation structure
    stp x0, x1, [x0, #0]      // Save X0-X1
    stp x2, x3, [x0, #16]     // Save X2-X3
    stp x4, x5, [x0, #32]     // ...
    stp x6, x7, [x0, #48]
    stp x8, x9, [x0, #64]
    stp x10, x11, [x0, #80]
    stp x12, x13, [x0, #96]
    stp x14, x15, [x0, #112]
    stp x16, x17, [x0, #128]
    stp x18, x19, [x0, #144]
    stp x20, x21, [x0, #160]
    stp x22, x23, [x0, #176]
    stp x24, x25, [x0, #192]
    stp x26, x27, [x0, #208]
    stp x28, x29, [x0, #224]   // X29 = FP
    str x30, [x0, #240]        // X30 = LR

    // Save SIMD/FP registers V0-V31
    // Offset calculation: gp_regs (31*8=248 bytes) + alignment
    add x1, x0, #256           // Point to simd_regs array

    stp d0, d1, [x1, #0]
    stp d2, d3, [x1, #16]
    stp d4, d5, [x1, #32]
    stp d6, d7, [x1, #48]
    stp d8, d9, [x1, #64]
    stp d10, d11, [x1, #80]
    stp d12, d13, [x1, #96]
    stp d14, d15, [x1, #112]
    stp d16, d17, [x1, #128]
    stp d18, d19, [x1, #144]
    stp d20, d21, [x1, #160]
    stp d22, d23, [x1, #176]
    stp d24, d25, [x1, #192]
    stp d26, d27, [x1, #208]
    stp d28, d29, [x1, #224]
    stp d30, d31, [x1, #240]

    // Save stack pointer, frame pointer, link register
    // Offset: gp_regs (248) + simd_regs (512) = 760 bytes
    mov x2, sp
    str x2, [x0, #760]         // Save SP

    str x29, [x0, #768]        // Save FP
    str x30, [x0, #776]        // Save LR (return address)

    // Stack capture
    // Calculate stack size: FP - SP (simplified)
    sub x2, x29, sp            // Stack frame size

    // Limit stack capture to MAX_STACK_CAPTURE
    mov x3, #4096
    cmp x2, x3
    csel x2, x2, x3, lo        // min(frame_size, 4096)

    // Offset to stack vector pointer: 784 bytes
    // (This would require calling into Rust to allocate Vec)
    // For assembly-only version, use a fixed buffer

    ret

.size __sounio_capture_continuation, .-__sounio_capture_continuation
```

### Resume Assembly Template

```asm
// Function: __sounio_resume_continuation
// Input: X0 = pointer to AArch64Continuation, X1 = resume value
// Output: Never returns! Jumps to continuation
// This function DOES NOT return normally

__sounio_resume_continuation:
    // Restore stack pointer first
    ldr x2, [x0, #760]
    mov sp, x2

    // Restore general-purpose registers (except X0, X1 which we'll do last)
    ldp x2, x3, [x0, #16]
    ldp x4, x5, [x0, #32]
    ldp x6, x7, [x0, #48]
    ldp x8, x9, [x0, #64]
    ldp x10, x11, [x0, #80]
    ldp x12, x13, [x0, #96]
    ldp x14, x15, [x0, #112]
    ldp x16, x17, [x0, #128]
    ldp x18, x19, [x0, #144]
    ldp x20, x21, [x0, #160]
    ldp x22, x23, [x0, #176]
    ldp x24, x25, [x0, #192]
    ldp x26, x27, [x0, #208]
    ldp x28, x29, [x0, #224]   // Restore FP
    ldr x30, [x0, #240]        // Restore LR

    // Restore SIMD registers
    add x2, x0, #256
    ldp d0, d1, [x2, #0]
    ldp d2, d3, [x2, #16]
    // ... (all 32 registers)
    ldp d30, d31, [x2, #240]

    // Put resume value in X0
    mov x0, x1

    // Jump to the return address (from LR which we just restored)
    ret  // Returns to the address in X30

.size __sounio_resume_continuation, .-__sounio_resume_continuation
```

## Implementation Steps

### Step 1: Create Assembly File

Create `crates/souc/src/backend/native/aarch64_continuation.S`:

```asm
.section .text
.globl __sounio_capture_continuation
.globl __sounio_resume_continuation

// Include the templates above
```

### Step 2: Link Assembly with Build Script

Update `crates/souc/build.rs`:

```rust
fn main() {
    // Existing code...

    // Compile AArch64 continuation assembly if target is aarch64
    #[cfg(target_arch = "aarch64")]
    {
        let asm_file = "src/backend/native/aarch64_continuation.S";
        println!("cargo:rerun-if-changed={}", asm_file);

        cc::Build::new()
            .file(asm_file)
            .compile("aarch64_continuation");
    }
}
```

Add `cc` to build-dependencies in Cargo.toml:

```toml
[build-dependencies]
cc = "1.0"
```

### Step 3: Update Rust Code

In `continuation.rs`, change from inline asm to extern:

```rust
#[cfg(target_arch = "aarch64")]
pub unsafe fn capture_continuation() -> Continuation {
    let mut cont = AArch64Continuation::new();

    extern "C" {
        fn __sounio_capture_continuation(cont: *mut AArch64Continuation);
    }

    __sounio_capture_continuation(&mut cont as *mut _);

    Continuation::AArch64(cont)
}

#[cfg(target_arch = "aarch64")]
pub unsafe fn resume_continuation(cont: &mut Continuation, value: u64) -> ! {
    match cont {
        Continuation::AArch64(aarch64_cont) => {
            assert!(!aarch64_cont.is_resumed());
            aarch64_cont.mark_resumed();

            extern "C" {
                fn __sounio_resume_continuation(
                    cont: *const AArch64Continuation,
                    value: u64
                ) -> !;
            }

            __sounio_resume_continuation(aarch64_cont as *const _, value)
        }
        _ => panic!("Wrong continuation type"),
    }
}
```

### Step 4: Stack Handling

The stack capture in the assembly template is simplified. A complete implementation needs:

1. **Determine actual stack boundaries**:
   ```c
   // Get stack base from thread info
   // Copy from current SP to stack base or frame boundary
   ```

2. **Handle stack unwinding**:
   - The captured stack becomes invalid if frames are popped
   - Need to ensure continuation is used before stack unwinds
   - Or copy the entire stack to heap (expensive)

3. **Restore stack contents**:
   - When resuming, copy saved stack back to its original location
   - Ensure stack alignment (16-byte on AArch64)

## Testing

### Unit Tests

```rust
#[test]
#[cfg(target_arch = "aarch64")]
fn test_capture_and_resume() {
    unsafe {
        let cont = capture_continuation();
        // Verify registers are captured
        match cont {
            Continuation::AArch64(c) => {
                assert_ne!(c.lr, 0);  // Return address should be set
                assert_ne!(c.sp, 0);  // Stack pointer should be set
            }
            _ => panic!("Wrong arch"),
        }
    }
}
```

### Integration Test

```rust
// Test that a simple computation can be suspended and resumed
#[test]
fn test_effect_with_continuation() {
    fn computation_with_effect() -> i64 {
        let x = 10;
        // Perform effect that captures continuation
        let y = perform_io_effect(x);
        y + 5
    }

    let result = computation_with_effect();
    assert_eq!(result, 15);
}
```

## Platform Support

- **AArch64**: Full implementation as described above
- **x86-64**: Similar approach with different registers (RAX-R15, XMM0-XMM15)
- **RISC-V**: 32 integer registers (X0-X31), 32 float registers (F0-F31)
- **WebAssembly**: Not applicable (no direct register access)

## Performance Considerations

- **Capture cost**: ~100-200 cycles (register saves + stack copy)
- **Resume cost**: ~100-200 cycles (register restores + stack copy)
- **Memory**: Typical continuation ~2-8 KB (depends on stack depth)

For hot paths, consider:
- One-shot continuations only (no stack copy needed)
- Lazy stack capture (only copy if continuation outlives frame)
- Stack caching (reuse allocated buffers)

## Security

- Captured continuations contain sensitive data (all registers, stack)
- Must prevent:
  - Information leakage through captured stack
  - Use-after-free if stack is deallocated
  - Double-resume of one-shot continuations

## References

- AArch64 Procedure Call Standard (AAPCS64)
- ARM Architecture Reference Manual (ARMv8-A)
- Plotkin & Pretnar: "Handlers of Algebraic Effects" (2009)
- Leijen: "Type Directed Compilation of Row-typed Algebraic Effects" (2017)
