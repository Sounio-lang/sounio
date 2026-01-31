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
//! - `RealAsyncHandler`: Real async handler using Tokio runtime
//! - `IOHandler`: Handles IO operations (print, read_file, write_file, etc.)
//! - `MutHandler`: Handles mutable state operations (get, set, modify, with_state, etc.)
//! - `PanicHandler`: Handles recoverable failures (panic, assert, unwrap, etc.)
//! - `ProbHandler`: Handles probabilistic effects (sample, observe, condition, etc.)
//! - `DivHandler`: Handles division operations (div, checked_div, safe_div, etc.)
//! - `GpuHandler`: Handles GPU operations (launch, sync, alloc_device, etc.)
//! - `RealGpuHandler`: Real GPU handler using actual GPU operations
//! - `BatchedGpuHandler`: Batching wrapper for GPU operations (reduces sync overhead)
//! - `EpistemicHandler`: Handles epistemic operations (degrade, assert_confidence, firewall, etc.)
//! - `NetworkHandler`: Handles network operations (fetch, post, websocket, etc.) - simulated
//! - `RealNetworkHandler`: Real network handler using Tokio for TCP/UDP/HTTP
//! - `SensorHandler`: Handles sensor operations (read, calibrate, batch_read, etc.)
//! - `ExnHandler`: Handles typed exception operations (throw, try_catch, rethrow, etc.)
//!
//! # Handler Registry
//!
//! Use `HandlerRegistry` to manage handlers centrally:
//!
//! ```ignore
//! use sounio::effects::handlers::{HandlerRegistry, DefaultHandlerFactory};
//!
//! // Create with all default handlers
//! let registry = HandlerRegistry::with_defaults();
//!
//! // Or build selectively
//! let registry = HandlerRegistryBuilder::new()
//!     .with_io()
//!     .with_mut()
//!     .build();
//! ```
//!
//! # Handler Composition
//!
//! Compose multiple handlers for multi-effect functions (`with IO, Mut, Epistemic`):
//!
//! ```ignore
//! use sounio::effects::handlers::{HandlerComposer, CompositionOrder};
//!
//! // Compose handlers with left-to-right dispatch order
//! let composed = HandlerComposer::new()
//!     .with(Arc::new(IOHandler::new()))
//!     .with(Arc::new(MutHandler::new()))
//!     .with(Arc::new(EpistemicHandler::new()))
//!     .order(CompositionOrder::LeftToRight)
//!     .build()?;
//! ```
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
mod async_handler_real;
mod batched_gpu_handler;
mod composition;
mod div_handler;
mod epistemic_handler;
mod exn_handler;
mod gpu_handler;
mod gpu_handler_real;
mod io_handler;
mod mut_handler;
mod network_handler;
mod network_handler_real;
mod panic_handler;
mod prob_handler;
mod registry;
mod sensor_handler;

pub use alloc_handler::AllocHandler;
pub use async_handler::AsyncHandler;
pub use async_handler_real::RealAsyncHandler;
pub use batched_gpu_handler::{BatchError, BatchedGpuHandler, BatchedOp};
pub use composition::{
    ComposeError, ComposedHandler, CompositionOrder, HandlerComposer, HandlerStack,
};
pub use div_handler::DivHandler;
pub use epistemic_handler::EpistemicHandler;
pub use exn_handler::ExnHandler;
pub use gpu_handler::GpuHandler;
pub use gpu_handler_real::RealGpuHandler;
pub use io_handler::IOHandler;
pub use mut_handler::MutHandler;
pub use network_handler::NetworkHandler;
pub use network_handler_real::RealNetworkHandler;
pub use panic_handler::PanicHandler;
pub use prob_handler::ProbHandler;
pub use registry::{
    DefaultHandlerFactory, HandlerFactory, HandlerRegistry, HandlerRegistryBuilder,
};
pub use sensor_handler::SensorHandler;
