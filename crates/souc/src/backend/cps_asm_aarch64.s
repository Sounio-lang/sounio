// AArch64 Assembly Stubs for Continuation Capture/Resume
//
// These functions provide low-level register saving and restoring for
// continuation capture on AArch64 architecture.
//
// Calling Convention (AAPCS64):
// - x0-x7: Argument/result registers
// - x8: Indirect result location
// - x9-x15: Temporary registers
// - x16-x17: Intra-procedure-call temporary (IP0, IP1)
// - x18: Platform register
// - x19-x28: Callee-saved registers
// - x29: Frame pointer (FP)
// - x30: Link register (LR)
// - sp: Stack pointer
//
// - v0-v7: Argument/result registers (SIMD/FP)
// - v8-v15: Callee-saved (lower 64 bits only)
// - v16-v31: Temporary registers

.global __sounio_capture_continuation_asm
.global __sounio_resume_continuation_asm
.global __sounio_get_return_address

.text
.align 4

// Capture current continuation state
//
// C signature: void* __sounio_capture_continuation_asm(void* cont_ptr)
//
// Arguments:
//   x0 = pointer to NativeContinuation struct
//
// Returns:
//   x0 = same pointer (for chaining)
//
// This function saves all registers into the continuation structure:
// - Offset 0: id (16 bytes)
// - Offset 16: return_address (8 bytes)
// - Offset 24: gp_registers[32] (256 bytes)
// - Offset 280: fp_registers[32] (256 bytes)
// - Offset 536: stack_pointer (8 bytes)
// - Offset 544: frame_pointer (8 bytes)
__sounio_capture_continuation_asm:
    // Save continuation pointer for later
    mov     x19, x0

    // Save return address (from LR)
    str     x30, [x19, #16]

    // Save general-purpose registers (x0-x30)
    // Note: We save x0-x30 sequentially at offset 24
    add     x20, x19, #24

    stp     x0, x1, [x20, #0]
    stp     x2, x3, [x20, #16]
    stp     x4, x5, [x20, #32]
    stp     x6, x7, [x20, #48]
    stp     x8, x9, [x20, #64]
    stp     x10, x11, [x20, #80]
    stp     x12, x13, [x20, #96]
    stp     x14, x15, [x20, #112]
    stp     x16, x17, [x20, #128]
    stp     x18, x19, [x20, #144]
    stp     x20, x21, [x20, #160]
    stp     x22, x23, [x20, #176]
    stp     x24, x25, [x20, #192]
    stp     x26, x27, [x20, #208]
    stp     x28, x29, [x20, #224]
    str     x30, [x20, #240]

    // Save floating-point/SIMD registers (v0-v31)
    // Each register is 128 bits (16 bytes), but we only save lower 64 bits (8 bytes)
    // Offset: 280
    add     x20, x19, #280

    stp     d0, d1, [x20, #0]
    stp     d2, d3, [x20, #16]
    stp     d4, d5, [x20, #32]
    stp     d6, d7, [x20, #48]
    stp     d8, d9, [x20, #64]
    stp     d10, d11, [x20, #80]
    stp     d12, d13, [x20, #96]
    stp     d14, d15, [x20, #112]
    stp     d16, d17, [x20, #128]
    stp     d18, d19, [x20, #144]
    stp     d20, d21, [x20, #160]
    stp     d22, d23, [x20, #176]
    stp     d24, d25, [x20, #192]
    stp     d26, d27, [x20, #208]
    stp     d28, d29, [x20, #224]
    stp     d30, d31, [x20, #240]

    // Save stack pointer
    // Offset: 536
    mov     x20, sp
    str     x20, [x19, #536]

    // Save frame pointer (x29)
    // Offset: 544
    str     x29, [x19, #544]

    // Return continuation pointer in x0
    mov     x0, x19
    ret

// Resume a continuation
//
// C signature: void __sounio_resume_continuation_asm(void* cont_ptr, u64 value) __attribute__((noreturn))
//
// Arguments:
//   x0 = pointer to NativeContinuation struct
//   x1 = resume value (will be placed in x0 after restoration)
//
// This function restores all registers from the continuation structure
// and jumps to the saved return address. It never returns.
__sounio_resume_continuation_asm:
    // Save resume value temporarily
    mov     x19, x1

    // Load continuation pointer
    mov     x20, x0

    // Restore frame pointer first (needed for stack operations)
    ldr     x29, [x20, #544]

    // Restore stack pointer
    ldr     x21, [x20, #536]
    mov     sp, x21

    // Restore floating-point/SIMD registers (v0-v31)
    add     x21, x20, #280

    ldp     d0, d1, [x21, #0]
    ldp     d2, d3, [x21, #16]
    ldp     d4, d5, [x21, #32]
    ldp     d6, d7, [x21, #48]
    ldp     d8, d9, [x21, #64]
    ldp     d10, d11, [x21, #80]
    ldp     d12, d13, [x21, #96]
    ldp     d14, d15, [x21, #112]
    ldp     d16, d17, [x21, #128]
    ldp     d18, d19, [x21, #144]
    ldp     d20, d21, [x21, #160]
    ldp     d22, d23, [x21, #176]
    ldp     d24, d25, [x21, #192]
    ldp     d26, d27, [x21, #208]
    ldp     d28, d29, [x21, #224]
    ldp     d30, d31, [x21, #240]

    // Restore general-purpose registers (x1-x30, skip x0 for now)
    add     x21, x20, #24

    // Skip x0 (will be set to resume value)
    ldp     x1, x2, [x21, #8]    // Load x1, x2 (skip x0)
    ldp     x3, x4, [x21, #24]
    ldp     x5, x6, [x21, #40]
    ldp     x7, x8, [x21, #56]
    ldp     x9, x10, [x21, #72]
    ldp     x11, x12, [x21, #88]
    ldp     x13, x14, [x21, #104]
    ldp     x15, x16, [x21, #120]
    ldp     x17, x18, [x21, #136]
    // Skip x19, x20 (in use)
    ldp     x21, x22, [x21, #168]
    ldp     x23, x24, [x21, #184]
    ldp     x25, x26, [x21, #200]
    ldp     x27, x28, [x21, #216]
    // x29 already restored
    ldr     x30, [x21, #240]     // Restore link register

    // Set x0 to resume value
    mov     x0, x19

    // Load return address
    ldr     x21, [x20, #16]

    // Jump to return address (continuation resumes here)
    br      x21

// Get the return address of the caller
//
// C signature: usize __sounio_get_return_address()
//
// Returns:
//   x0 = return address (value of LR)
__sounio_get_return_address:
    mov     x0, x30
    ret

.size __sounio_capture_continuation_asm, . - __sounio_capture_continuation_asm
.size __sounio_resume_continuation_asm, . - __sounio_resume_continuation_asm
.size __sounio_get_return_address, . - __sounio_get_return_address

.section .note.GNU-stack,"",@progbits
