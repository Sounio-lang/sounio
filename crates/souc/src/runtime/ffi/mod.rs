//! FFI Functions for Sounio Self-Hosting
//!
//! This module contains all 44 FFI functions that Sounio code calls at runtime.
//! These functions provide the interface between Sounio bytecode and Rust runtime.

pub mod ffi_alloc;
pub mod ffi_io;
pub mod ffi_path;
pub mod ffi_process;
pub mod ffi_stdio;
pub mod ffi_time;

pub use ffi_alloc::*;
pub use ffi_io::*;
pub use ffi_path::*;
pub use ffi_process::*;
pub use ffi_stdio::*;
pub use ffi_time::*;
