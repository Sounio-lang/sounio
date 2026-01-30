//! Continuation Capture and Resumption for Native Backend
//!
//! This module implements one-shot continuation capture and resumption for the
//! native backend (AArch64 and x86-64).
//!
//! # Theory
//!
//! When an effect operation is performed, we need to capture the current
//! execution state (continuation) so the handler can decide whether and when
//! to resume it. For native code, this requires:
//!
//! 1. **Register State**: Save all general-purpose and SIMD registers
//! 2. **Stack State**: Capture the current stack frame and parent frames
//! 3. **Return Address**: Where to resume execution
//!
//! # AArch64 Continuation Capture
//!
//! For AArch64 (ARM64), we need to save:
//! - X0-X30: General-purpose registers (64-bit)
//! - V0-V31: SIMD/FP registers (128-bit each, store as pairs of u64)
//! - SP: Stack pointer
//! - FP: Frame pointer (X29)
//! - LR: Link register (X30, contains return address)
//!
//! The stack is captured from the current SP up to a safe limit.
//!
//! # One-Shot vs Multi-Shot
//!
//! Currently we only implement one-shot continuations (can be resumed once).
//! Multi-shot continuations require deep copying the entire stack, which is
//! expensive and rarely needed.

use std::fmt;

/// Maximum stack size to capture (2MB)
const MAX_STACK_CAPTURE: usize = 2 * 1024 * 1024;

/// AArch64 continuation state
///
/// This structure holds all the machine state needed to resume execution
/// at a continuation point.
#[repr(C)]
#[derive(Clone)]
pub struct AArch64Continuation {
    /// General-purpose registers X0-X30 (31 registers)
    pub gp_regs: [u64; 31],

    /// SIMD/FP registers V0-V31 (32 registers, 128-bit each stored as 2x u64)
    /// Layout: [V0_low, V0_high, V1_low, V1_high, ..., V31_low, V31_high]
    pub simd_regs: [u64; 64],

    /// Stack pointer value at capture time
    pub sp: u64,

    /// Frame pointer value at capture time (same as X29)
    pub fp: u64,

    /// Link register value (return address, same as X30)
    pub lr: u64,

    /// Captured stack contents (from SP upward)
    pub stack: Vec<u8>,

    /// Whether this continuation has been resumed (one-shot enforcement)
    pub resumed: bool,
}

impl AArch64Continuation {
    /// Create a new empty continuation
    pub fn new() -> Self {
        Self {
            gp_regs: [0; 31],
            simd_regs: [0; 64],
            sp: 0,
            fp: 0,
            lr: 0,
            stack: Vec::new(),
            resumed: false,
        }
    }

    /// Mark this continuation as resumed
    pub fn mark_resumed(&mut self) {
        self.resumed = true;
    }

    /// Check if this continuation has been resumed
    pub fn is_resumed(&self) -> bool {
        self.resumed
    }

    /// Get the return address (where to resume execution)
    pub fn return_address(&self) -> usize {
        self.lr as usize
    }

    /// Get the stack size in bytes
    pub fn stack_size(&self) -> usize {
        self.stack.len()
    }
}

impl Default for AArch64Continuation {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for AArch64Continuation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AArch64Continuation")
            .field("sp", &format_args!("{:#x}", self.sp))
            .field("fp", &format_args!("{:#x}", self.fp))
            .field("lr", &format_args!("{:#x}", self.lr))
            .field("stack_size", &self.stack.len())
            .field("resumed", &self.resumed)
            .finish()
    }
}

/// x86-64 continuation state (placeholder for future implementation)
#[repr(C)]
#[derive(Clone, Debug)]
pub struct X86_64Continuation {
    /// General-purpose registers RAX, RBX, RCX, RDX, RSI, RDI, R8-R15
    pub gp_regs: [u64; 14],

    /// SSE registers XMM0-XMM15 (128-bit each stored as 2x u64)
    pub sse_regs: [u64; 32],

    /// Stack pointer
    pub rsp: u64,

    /// Base pointer
    pub rbp: u64,

    /// Return address
    pub rip: u64,

    /// Captured stack
    pub stack: Vec<u8>,

    /// One-shot enforcement
    pub resumed: bool,
}

impl Default for X86_64Continuation {
    fn default() -> Self {
        Self {
            gp_regs: [0; 14],
            sse_regs: [0; 32],
            rsp: 0,
            rbp: 0,
            rip: 0,
            stack: Vec::new(),
            resumed: false,
        }
    }
}

/// Platform-independent continuation
#[derive(Clone, Debug)]
pub enum Continuation {
    /// AArch64 continuation
    AArch64(AArch64Continuation),

    /// x86-64 continuation
    X86_64(X86_64Continuation),
}

impl Continuation {
    /// Create a new AArch64 continuation
    pub fn aarch64() -> Self {
        Continuation::AArch64(AArch64Continuation::new())
    }

    /// Create a new x86-64 continuation
    pub fn x86_64() -> Self {
        Continuation::X86_64(X86_64Continuation::default())
    }

    /// Check if this continuation has been resumed
    pub fn is_resumed(&self) -> bool {
        match self {
            Continuation::AArch64(cont) => cont.is_resumed(),
            Continuation::X86_64(cont) => cont.resumed,
        }
    }

    /// Mark this continuation as resumed
    pub fn mark_resumed(&mut self) {
        match self {
            Continuation::AArch64(cont) => cont.mark_resumed(),
            Continuation::X86_64(cont) => cont.resumed = true,
        }
    }

    /// Get the return address
    pub fn return_address(&self) -> usize {
        match self {
            Continuation::AArch64(cont) => cont.return_address(),
            Continuation::X86_64(cont) => cont.rip as usize,
        }
    }
}

/// Capture the current continuation (platform-specific)
///
/// # Current Implementation Status
///
/// This is a simplified implementation that captures minimal state for testing.
/// A full implementation would need to be written in assembly or use compiler
/// intrinsics to properly save all registers without clobbering them.
///
/// # What a Full AArch64 Implementation Needs
///
/// 1. **Register Capture** (assembly required):
///    - Save X0-X30 using STP (store pair) instructions
///    - Save V0-V31 using STP with D registers
///    - Capture SP, FP (X29), LR (X30) before any function calls
///
/// 2. **Stack Capture**:
///    - Calculate actual stack frame boundaries
///    - Copy stack contents from current SP to frame base
///    - Handle stack alignment requirements (16-byte on AArch64)
///
/// 3. **Return Address**:
///    - Capture the link register (X30) which contains the return address
///    - Adjust for any trampolines or wrapper functions
///
/// # Safety
///
/// This function must be called from Rust code with a valid stack frame.
/// The captured stack may become invalid if the original stack is unwound.
#[cfg(target_arch = "aarch64")]
pub unsafe fn capture_continuation() -> Continuation {
    let mut cont = AArch64Continuation::new();

    extern "C" {
        fn __sounio_capture_aarch64(cont: *mut AArch64Continuation);
    }

    __sounio_capture_aarch64(&mut cont as *mut _);

    Continuation::AArch64(cont)
}

/// Resume a continuation (platform-specific)
///
/// This function restores the saved register and stack state, then jumps
/// to the return address to resume execution.
///
/// # Safety
///
/// This function:
/// - Never returns (it jumps to the continuation)
/// - Invalidates the current stack frame
/// - Must only be called with a valid, non-resumed continuation
///
/// # Arguments
///
/// - `cont`: The continuation to resume
/// - `value`: The value to pass to the resumed computation (passed in X0/RAX)
#[cfg(target_arch = "aarch64")]
pub unsafe fn resume_continuation(cont: &mut Continuation, value: u64) -> ! {
    match cont {
        Continuation::AArch64(aarch64_cont) => {
            assert!(
                !aarch64_cont.is_resumed(),
                "Attempted to resume one-shot continuation twice"
            );
            aarch64_cont.mark_resumed();

            extern "C" {
                fn __sounio_resume_aarch64(cont: *const AArch64Continuation, value: u64) -> !;
            }

            __sounio_resume_aarch64(aarch64_cont as *const _, value)
        }
        _ => panic!("Wrong continuation type for AArch64"),
    }
}

/// Stub implementation for non-AArch64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn capture_continuation() -> Continuation {
    panic!("Continuation capture not implemented for this platform");
}

/// Stub implementation for non-AArch64 platforms
#[cfg(not(target_arch = "aarch64"))]
pub unsafe fn resume_continuation(_cont: &mut Continuation, _value: u64) -> ! {
    panic!("Continuation resume not implemented for this platform");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continuation_creation() {
        let cont = AArch64Continuation::new();
        assert_eq!(cont.sp, 0);
        assert_eq!(cont.fp, 0);
        assert_eq!(cont.lr, 0);
        assert!(!cont.is_resumed());
    }

    #[test]
    fn test_continuation_one_shot() {
        let mut cont = AArch64Continuation::new();
        assert!(!cont.is_resumed());

        cont.mark_resumed();
        assert!(cont.is_resumed());
    }

    #[test]
    fn test_platform_continuation() {
        let cont = Continuation::aarch64();
        assert!(!cont.is_resumed());
        assert_eq!(cont.return_address(), 0);
    }
}
