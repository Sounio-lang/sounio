//! FFI Functions for Sounio Self-Hosting
//!
//! This module contains all FFI functions that Sounio code calls at runtime.
//! These functions provide the interface between Sounio bytecode and Rust runtime.

pub mod ffi_alloc;
#[cfg(feature = "blas")]
pub mod ffi_blas;
pub mod ffi_chiuratto;
#[cfg(feature = "hdf5")]
pub mod ffi_hdf5;
#[cfg(feature = "parquet")]
pub mod ffi_parquet;
pub mod ffi_compress;
pub mod ffi_gpu;
pub mod ffi_io;
pub mod ffi_path;
pub mod ffi_process;
pub mod ffi_stdio;
pub mod ffi_time;

pub use ffi_alloc::*;
pub use ffi_chiuratto::*;
pub use ffi_compress::*;
pub use ffi_gpu::*;
pub use ffi_io::*;
#[cfg(feature = "parquet")]
pub use ffi_parquet::*;
pub use ffi_path::*;
pub use ffi_process::*;
pub use ffi_stdio::*;
pub use ffi_time::*;
