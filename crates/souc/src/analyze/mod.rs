//! Code analysis tools for Sounio
//!
//! Provides:
//! - Code metrics (LOC, complexity, etc.)
//! - Dead code detection
//! - Dependency analysis
//! - Code smell detection

pub mod dead_code;
pub mod metrics;

pub use dead_code::{analyze_dead_code, DeadCodeReport, UnreachableCode, UnusedItem};
pub use metrics::{calculate_metrics, FileMetrics, FunctionMetrics};
