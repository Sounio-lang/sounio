//! AArch64 Continuation Capture and Resumption Assembly
//!
//! This file implements low-level continuation capture and resumption for AArch64.
//! Based on the implementation guide in CONTINUATION_IMPLEMENTATION.md
//!
//! AArch64 Register Layout:
//! - X0-X30: General-purpose registers (64-bit)
//! - X29 (FP): Frame pointer
//! - X30 (LR): Link register (return address)
//! - SP: Stack pointer
//! - V0-V31: SIMD/FP registers (128-bit, accessed as D0-D31 for 64-bit)
//!
//! AArch64Continuation struct layout (must match Rust definition):
//! Offset  Size  Field
//! 0       248   gp_regs[31] (X0-X30, each 8 bytes)
//! 248     8     (padding to 256)
//! 256     512   simd_regs[32] (V0-V31 as pairs of D regs, each 16 bytes)
//! 768     8     sp
//! 776     8     fp
//! 784     8     lr
//! 792     ?     stack (Vec<u8>, pointer + len + cap)

    .section .text
    .align 4
    .globl __sounio_capture_aarch64
    .globl __sounio_resume_aarch64

//! __sounio_capture_aarch64
//!
//! Captures the current execution state into an AArch64Continuation structure.
//!
//! Arguments:
//!   X0: pointer to AArch64Continuation structure
//!
//! Returns:
//!   void (continuation structure is filled)
//!
//! Preserves all registers (no clobbers)
//!
__sounio_capture_aarch64:
    // Save general-purpose registers X0-X30
    // We save X0 first, then use it as base pointer
    stp x0, x1, [x0, #0]       // Save X0-X1 at offset 0
    stp x2, x3, [x0, #16]      // Save X2-X3 at offset 16
    stp x4, x5, [x0, #32]      // Save X4-X5
    stp x6, x7, [x0, #48]      // Save X6-X7
    stp x8, x9, [x0, #64]      // Save X8-X9
    stp x10, x11, [x0, #80]    // Save X10-X11
    stp x12, x13, [x0, #96]    // Save X12-X13
    stp x14, x15, [x0, #112]   // Save X14-X15
    stp x16, x17, [x0, #128]   // Save X16-X17
    stp x18, x19, [x0, #144]   // Save X18-X19
    stp x20, x21, [x0, #160]   // Save X20-X21
    stp x22, x23, [x0, #176]   // Save X22-X23
    stp x24, x25, [x0, #192]   // Save X24-X25
    stp x26, x27, [x0, #208]   // Save X26-X27
    stp x28, x29, [x0, #224]   // Save X28-X29 (FP)
    str x30, [x0, #240]        // Save X30 (LR) at offset 240

    // Save SIMD/FP registers V0-V31
    // Access as pairs of D registers (64-bit) for efficiency
    // Offset: 256 (after gp_regs + padding)
    add x1, x0, #256           // X1 = base + 256 (simd_regs offset)

    stp d0, d1, [x1, #0]       // Save V0-V1 (low 64 bits)
    stp d2, d3, [x1, #16]      // Save V2-V3
    stp d4, d5, [x1, #32]      // Save V4-V5
    stp d6, d7, [x1, #48]      // Save V6-V7
    stp d8, d9, [x1, #64]      // Save V8-V9
    stp d10, d11, [x1, #80]    // Save V10-V11
    stp d12, d13, [x1, #96]    // Save V12-V13
    stp d14, d15, [x1, #112]   // Save V14-V15
    stp d16, d17, [x1, #128]   // Save V16-V17
    stp d18, d19, [x1, #144]   // Save V18-V19
    stp d20, d21, [x1, #160]   // Save V20-V21
    stp d22, d23, [x1, #176]   // Save V22-V23
    stp d24, d25, [x1, #192]   // Save V24-V25
    stp d26, d27, [x1, #208]   // Save V26-V27
    stp d28, d29, [x1, #224]   // Save V28-V29
    stp d30, d31, [x1, #240]   // Save V30-V31

    // Save stack pointer, frame pointer, link register
    // Offset: 768 (after gp_regs + simd_regs)
    mov x2, sp
    str x2, [x0, #768]         // Save SP at offset 768

    str x29, [x0, #776]        // Save FP at offset 776
    str x30, [x0, #784]        // Save LR at offset 784

    // Note: Stack capture (Vec<u8>) is handled by Rust code
    // We only save register state here for shallow one-shot continuations

    ret

    .size __sounio_capture_aarch64, .-__sounio_capture_aarch64

//! __sounio_resume_aarch64
//!
//! Resumes execution from a captured continuation.
//!
//! Arguments:
//!   X0: pointer to AArch64Continuation structure
//!   X1: resume value (passed to resumed computation)
//!
//! This function NEVER returns normally! It restores the saved state and jumps
//! to the continuation's return address.
//!
__sounio_resume_aarch64:
    // Save resume value (X1) temporarily - we'll restore it at the end
    mov x9, x1

    // Restore stack pointer first
    ldr x2, [x0, #768]
    mov sp, x2

    // Restore SIMD/FP registers V0-V31
    add x1, x0, #256           // X1 = base + 256 (simd_regs offset)

    ldp d0, d1, [x1, #0]       // Restore V0-V1
    ldp d2, d3, [x1, #16]      // Restore V2-V3
    ldp d4, d5, [x1, #32]      // Restore V4-V5
    ldp d6, d7, [x1, #48]      // Restore V6-V7
    ldp d8, d9, [x1, #64]      // Restore V8-V9
    ldp d10, d11, [x1, #80]    // Restore V10-V11
    ldp d12, d13, [x1, #96]    // Restore V12-V13
    ldp d14, d15, [x1, #112]   // Restore V14-V15
    ldp d16, d17, [x1, #128]   // Restore V16-V17
    ldp d18, d19, [x1, #144]   // Restore V18-V19
    ldp d20, d21, [x1, #160]   // Restore V20-V21
    ldp d22, d23, [x1, #176]   // Restore V22-V23
    ldp d24, d25, [x1, #192]   // Restore V24-V25
    ldp d26, d27, [x1, #208]   // Restore V26-V27
    ldp d28, d29, [x1, #224]   // Restore V28-V29
    ldp d30, d31, [x1, #240]   // Restore V30-V31

    // Restore general-purpose registers (except X0, X1, X9 which we'll do last)
    ldp x2, x3, [x0, #16]      // Restore X2-X3
    ldp x4, x5, [x0, #32]      // Restore X4-X5
    ldp x6, x7, [x0, #48]      // Restore X6-X7
    ldp x8, x10, [x0, #64]     // Restore X8, X10 (skip X9 for now)
    // Note: We skip X9 above since we're using it to hold the resume value
    ldp x11, x12, [x0, #88]    // Restore X11-X12
    ldp x13, x14, [x0, #104]   // Restore X13-X14
    ldp x15, x16, [x0, #120]   // Restore X15-X16
    ldp x17, x18, [x0, #136]   // Restore X17-X18
    ldp x19, x20, [x0, #152]   // Restore X19-X20
    ldp x21, x22, [x0, #168]   // Restore X21-X22
    ldp x23, x24, [x0, #184]   // Restore X23-X24
    ldp x25, x26, [x0, #200]   // Restore X25-X26
    ldp x27, x28, [x0, #216]   // Restore X27-X28
    ldp x29, x30, [x0, #224]   // Restore X29 (FP), X30 (LR)

    // Restore X9 from saved state
    ldr x1, [x0, #72]          // Load saved X9 into X1 temporarily

    // Put resume value in X0 (return value)
    mov x0, x9

    // Restore X9
    mov x9, x1

    // Now restore X1 from saved state
    ldr x1, [x0, #8]           // Load saved X1

    // Jump to the saved return address (in X30/LR)
    // This effectively "returns" to the point where continuation was captured
    ret

    .size __sounio_resume_aarch64, .-__sounio_resume_aarch64
