//! Code analysis tools for Sounio
//!
//! Provides:
//! - Code metrics (LOC, complexity, etc.)
//! - Dead code detection
//! - Dependency analysis
//! - Code smell detection

pub mod dead_code;
pub mod metrics;

pub use dead_code::{DeadCodeReport, UnreachableCode, UnusedItem, analyze_dead_code};
pub use metrics::{FileMetrics, FunctionMetrics, calculate_metrics};
