//! Source maps for debugging and error reporting
//!
//! Tracks source locations through all compilation phases, enabling:
//! - Rich error messages with source context
//! - Debug info generation for compiled code
//! - Source-level debugging support

pub mod files;
pub mod location;
pub mod mapping;

pub use files::{FileId, SourceDb, SourceFile};
pub use location::{Located, SourceLocation, Span};
pub use mapping::{DebugInfo, SourceMap};
