//! Sounio Runtime Library
//!
//! This crate provides the runtime support for compiled Sounio programs,
//! including memory management, intrinsics, and standard library support.

pub mod memory;
pub mod intrinsics;

pub use memory::{Allocator, Arena, RcBox, SystemAllocator};
pub use intrinsics::Knowledge;
