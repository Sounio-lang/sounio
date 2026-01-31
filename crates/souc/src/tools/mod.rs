//! Sounio compiler tools
//!
//! This module contains various tools that work with the compiler:
//! - `pybindgen`: Python ctypes binding generator for FFI
//!
//! # Python Binding Generator
//!
//! Generate Python ctypes bindings from compiled Sounio libraries:
//!
//! ```ignore
//! use souc::tools::pybindgen::PythonBindingGenerator;
//! use souc::hir::Hir;
//!
//! let generator = PythonBindingGenerator::new();
//! let python_code = generator.generate_from_hir(&hir, "libexample.so")?;
//! ```

pub mod pybindgen;

pub use pybindgen::{generate_python_bindings, PythonBindingGenerator};
