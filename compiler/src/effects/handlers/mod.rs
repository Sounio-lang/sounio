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
//! - `IOHandler`: Handles IO operations (print, read_file, write_file, etc.)
//! - `MutHandler`: Handles mutable state operations (get, set, modify, with_state, etc.)
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
mod io_handler;
mod mut_handler;

pub use alloc_handler::AllocHandler;
pub use io_handler::IOHandler;
pub use mut_handler::MutHandler;
