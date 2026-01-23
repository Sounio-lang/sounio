//! Mixed-Precision Training Support for Sounio GPU Codegen
//!
//! Implements FP16 forward pass with FP32 backward pass for 2x memory bandwidth reduction.
//! Includes dynamic loss scaling to prevent gradient underflow in FP16.

mod config;
mod loss_scaler;

pub use config::{LowPrecision, MixedPrecisionConfig};
pub use loss_scaler::{LossScaler, ScaleUpdate};
