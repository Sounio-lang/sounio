//! Sounio Runtime Library
//!
//! This crate provides the runtime support for compiled Sounio programs,
//! including memory management, intrinsics, and standard library support.

pub mod handler_stack;
pub mod intrinsics;
pub mod memory;

pub use intrinsics::Knowledge;
pub use memory::{Allocator, Arena, RcBox, SystemAllocator};
