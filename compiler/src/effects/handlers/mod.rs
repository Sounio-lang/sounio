//! Effect handler implementations using the HandlerCapability trait
//!
//! This module contains concrete handler implementations for Sounio's algebraic
//! effects. Each handler implements the `HandlerCapability` trait, enabling:
//!
//! - Track A: Foundational handlers with continuation-passing style (CPS) semantics
//! - Track B: Epistemic handlers with confidence tracking and provenance
//!
//! # Available Handlers
//!
//! - `AllocHandler`: Handles memory allocation (alloc, dealloc, realloc, etc.)
//! - `AsyncHandler`: Handles async operations (spawn, await, join, select, etc.)
//! - `IOHandler`: Handles IO operations (print, read_file, write_file, etc.)
//! - `MutHandler`: Handles mutable state operations (get, set, modify, with_state, etc.)
//! - `PanicHandler`: Handles recoverable failures (panic, assert, unwrap, etc.)
//! - `ProbHandler`: Handles probabilistic effects (sample, observe, condition, etc.)
//! - `DivHandler`: Handles division operations (div, checked_div, safe_div, etc.)
//! - `GpuHandler`: Handles GPU operations (launch, sync, alloc_device, etc.)
//!
//! # Example
//!
//! ```ignore
//! use sounio::effects::handlers::{IOHandler, MutHandler};
//! use sounio::effects::handler_capability::HandlerCapability;
//!
//! let io_handler = IOHandler::new();
//! assert_eq!(io_handler.effect_name(), "IO");
//!
//! let mut_handler = MutHandler::new();
//! assert_eq!(mut_handler.effect_name(), "Mut");
//! ```

mod alloc_handler;
mod async_handler;
mod div_handler;
mod gpu_handler;
mod io_handler;
mod mut_handler;
mod panic_handler;
mod prob_handler;

pub use alloc_handler::AllocHandler;
pub use async_handler::AsyncHandler;
pub use div_handler::DivHandler;
pub use gpu_handler::GpuHandler;
pub use io_handler::IOHandler;
pub use mut_handler::MutHandler;
pub use panic_handler::PanicHandler;
pub use prob_handler::ProbHandler;
